#!/usr/bin/env python3
"""
Combined Multi-Agent System with Qdrant Vector Database
A comprehensive single-file implementation combining multi-agent coordination and vector storage
"""

import asyncio
import logging
import os
import json
import uuid
import numpy as np
from typing import Any, Dict, List, Optional, Union, TypedDict, Annotated, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod

# Core Dependencies
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition,
    MatchValue, CollectionInfo, UpdateResult, SearchRequest,
    CreateCollection, UpdateCollection
)
from qdrant_client.http.exceptions import ResponseHandlingException
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# ReAct Framework Components
# ============================================================================

class ActionType(Enum):
    SEARCH = "search"
    ANALYZE = "analyze"
    EXECUTE = "execute"
    COMMUNICATE = "communicate"
    PLAN = "plan"
    VALIDATE = "validate"

@dataclass
class Thought:
    content: str
    reasoning: str
    confidence: float
    timestamp: str

@dataclass
class Action:
    type: ActionType
    description: str
    parameters: Dict[str, Any]
    expected_outcome: str

@dataclass
class Observation:
    result: Any
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ReActAgent(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.memory: List[Tuple[Thought, Action, Observation]] = []
        self.max_iterations = 10

    @abstractmethod
    async def think(self, context: Dict[str, Any]) -> Thought:
        pass

    @abstractmethod
    async def act(self, thought: Thought, context: Dict[str, Any]) -> Action:
        pass

    @abstractmethod
    async def observe(self, action: Action) -> Observation:
        pass

    async def reason_act_cycle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the ReAct cycle: Reason -> Act -> Observe"""
        results = []

        for iteration in range(self.max_iterations):
            try:
                # Think (Reason)
                thought = await self.think(context)

                # Act
                action = await self.act(thought, context)

                # Observe
                observation = await self.observe(action)

                # Store in memory
                self.memory.append((thought, action, observation))

                results.append({
                    "iteration": iteration + 1,
                    "thought": thought,
                    "action": action,
                    "observation": observation
                })

                # Update context for next iteration
                context["previous_results"] = results

                # Check if task is complete
                if observation.success and "complete" in str(observation.result).lower():
                    break

            except Exception as e:
                logger.error(f"Error in ReAct cycle: {str(e)}")
                results.append({
                    "iteration": iteration + 1,
                    "error": str(e)
                })
                break

        return {
            "success": len(results) > 0 and results[-1].get("observation", {}).success,
            "iterations": len(results),
            "final_context": context,
            "results": results,
            "recommendations": self._generate_recommendations(results)
        }

    def _generate_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on execution results"""
        recommendations = []

        if len(results) > 5:
            recommendations.append("Consider breaking down complex tasks into smaller components")

        success_rate = sum(1 for r in results if r.get("observation", {}).success) / len(results)
        if success_rate < 0.7:
            recommendations.append("Review task complexity and agent capabilities")

        return recommendations

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of agent's memory"""
        return {
            "total_interactions": len(self.memory),
            "success_rate": sum(1 for _, _, obs in self.memory if obs.success) / len(self.memory) if self.memory else 0,
            "recent_actions": [action.type.value for _, action, _ in self.memory[-5:]],
        }

# ============================================================================
# Qdrant Vector Store Manager
# ============================================================================

@dataclass
class DocumentMetadata:
    agent_name: str
    document_type: str
    timestamp: str
    task_id: Optional[str] = None
    project_id: Optional[str] = None
    tags: List[str] = None
    priority: int = 1
    source: str = "multi_agent_system"

class QdrantVectorStore:
    def __init__(self,
                 host: str = "localhost",
                 port: int = 6333,
                 api_key: Optional[str] = None,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize Qdrant Vector Store Manager"""
        self.client = QdrantClient(host=host, port=port, api_key=api_key)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.vector_size = self.embedding_model.get_sentence_embedding_dimension()

        # Collection configurations
        self.collections = {
            "agent_knowledge": {
                "description": "General knowledge base for all agents",
                "vector_size": self.vector_size,
                "distance": Distance.COSINE
            },
            "qa_knowledge": {
                "description": "QA-specific knowledge and test results",
                "vector_size": self.vector_size,
                "distance": Distance.COSINE
            },
            "scheduler_knowledge": {
                "description": "Scheduling patterns and optimization data",
                "vector_size": self.vector_size,
                "distance": Distance.COSINE
            },
            "project_knowledge": {
                "description": "Project management insights and patterns",
                "vector_size": self.vector_size,
                "distance": Distance.COSINE
            },
            "coordination_logs": {
                "description": "Multi-agent coordination history and patterns",
                "vector_size": self.vector_size,
                "distance": Distance.COSINE
            }
        }

    async def initialize_collections(self) -> Dict[str, bool]:
        """Initialize all required collections"""
        results = {}

        for collection_name, config in self.collections.items():
            try:
                collections = self.client.get_collections()
                existing_collections = [col.name for col in collections.collections]

                if collection_name not in existing_collections:
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=config["vector_size"],
                            distance=config["distance"]
                        )
                    )
                    logger.info(f"Created collection: {collection_name}")

                results[collection_name] = True

            except Exception as e:
                logger.error(f"Failed to create collection {collection_name}: {str(e)}")
                results[collection_name] = False

        return results

    def create_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Create embeddings for given text(s)"""
        if isinstance(texts, str):
            texts = [texts]
        return self.embedding_model.encode(texts)

    async def store_agent_knowledge(self,
                                  agent_name: str,
                                  content: str,
                                  metadata: DocumentMetadata,
                                  collection_name: Optional[str] = None) -> str:
        """Store agent knowledge in vector database"""

        if collection_name is None:
            collection_name = f"{agent_name.lower()}_knowledge"
            if collection_name not in self.collections:
                collection_name = "agent_knowledge"

        # Create embedding
        embedding = self.create_embeddings(content)[0]
        point_id = str(uuid.uuid4())

        # Prepare payload
        payload = {
            "content": content,
            "agent_name": metadata.agent_name,
            "document_type": metadata.document_type,
            "timestamp": metadata.timestamp,
            "task_id": metadata.task_id,
            "project_id": metadata.project_id,
            "tags": metadata.tags or [],
            "priority": metadata.priority,
            "source": metadata.source
        }

        # Create point
        point = PointStruct(
            id=point_id,
            vector=embedding.tolist(),
            payload=payload
        )

        try:
            result = self.client.upsert(
                collection_name=collection_name,
                points=[point]
            )
            logger.info(f"Stored knowledge for {agent_name} in {collection_name}")
            return point_id
        except Exception as e:
            logger.error(f"Failed to store knowledge: {str(e)}")
            raise

    async def search_similar_knowledge(self,
                                     query: str,
                                     collection_name: str = "agent_knowledge",
                                     limit: int = 5,
                                     score_threshold: float = 0.7,
                                     filter_conditions: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar knowledge in vector database"""

        query_embedding = self.create_embeddings(query)[0]

        # Prepare filter if provided
        query_filter = None
        if filter_conditions:
            conditions = []
            for field, value in filter_conditions.items():
                conditions.append(
                    FieldCondition(
                        key=field,
                        match=MatchValue(value=value)
                    )
                )
            query_filter = Filter(must=conditions)

        try:
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding.tolist(),
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter
            )

            results = []
            for scored_point in search_result:
                result = {
                    "id": scored_point.id,
                    "score": scored_point.score,
                    "content": scored_point.payload.get("content", ""),
                    "agent_name": scored_point.payload.get("agent_name", ""),
                    "document_type": scored_point.payload.get("document_type", ""),
                    "timestamp": scored_point.payload.get("timestamp", ""),
                    "metadata": scored_point.payload
                }
                results.append(result)

            return results
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

# ============================================================================
# Agent Implementations
# ============================================================================

class QAAgent(ReActAgent):
    def __init__(self, llm_model: str = "gemini-1.5-flash"):
        super().__init__(
            name="QA_Agent",
            description="Quality Assurance agent responsible for testing, validation, and quality checks"
        )
        try:
            self.llm = ChatGoogleGenerativeAI(model=llm_model, temperature=0.1)
        except Exception:
            self.llm = None  # Fallback for demo mode

    async def think(self, context: Dict[str, Any]) -> Thought:
        current_task = context.get("task", "")
        previous_observations = [obs.result for _, _, obs in self.memory if obs.success]

        prompt = f"""
        As a QA Agent, analyze the current situation:

        Current Task: {current_task}
        Previous Observations: {previous_observations}
        Context: {context}

        What should I focus on for quality assurance? Consider:
        1. Testing requirements
        2. Quality standards
        3. Risk assessment
        4. Validation needs

        Provide your reasoning and confidence level (0-1).
        """

        if self.llm:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            content = response.content
        else:
            # Fallback response for demo mode
            content = f"As QA Agent, I will focus on comprehensive testing and quality validation for: {current_task}. I recommend implementing unit tests, integration tests, and performance validation."

        return Thought(
            content=content,
            reasoning="Analyzing QA requirements and quality standards",
            confidence=0.8,
            timestamp=datetime.now().isoformat()
        )

    async def act(self, thought: Thought, context: Dict[str, Any]) -> Action:
        if "test" in thought.content.lower():
            return Action(
                type=ActionType.EXECUTE,
                description="Execute comprehensive testing suite",
                parameters={
                    "test_type": "comprehensive",
                    "coverage_target": 0.85,
                    "quality_gates": ["unit_tests", "integration_tests", "performance_tests"]
                },
                expected_outcome="Test results with pass/fail status and quality metrics"
            )
        elif "validate" in thought.content.lower():
            return Action(
                type=ActionType.VALIDATE,
                description="Validate code quality and standards compliance",
                parameters={
                    "validation_rules": ["code_standards", "security_checks", "performance_metrics"],
                    "threshold": 0.9
                },
                expected_outcome="Validation report with compliance status"
            )
        else:
            return Action(
                type=ActionType.PLAN,
                description="Create QA strategy and test plan",
                parameters={
                    "strategy_focus": "comprehensive_quality_assurance",
                    "deliverables": ["test_plan", "quality_criteria", "acceptance_criteria"]
                },
                expected_outcome="Detailed QA plan and strategy document"
            )

    async def observe(self, action: Action) -> Observation:
        try:
            # Simulate QA action execution
            if action.type == ActionType.EXECUTE:
                result = {
                    "test_results": {
                        "unit_tests": {"passed": 95, "failed": 5, "coverage": 0.87},
                        "integration_tests": {"passed": 20, "failed": 2, "coverage": 0.82},
                        "performance_tests": {"passed": 8, "failed": 1, "avg_response_time": "120ms"}
                    },
                    "overall_status": "PASSED_WITH_WARNINGS",
                    "quality_score": 0.85
                }
            elif action.type == ActionType.VALIDATE:
                result = {
                    "code_standards": "PASSED",
                    "security_checks": "PASSED",
                    "performance_metrics": "WARNING",
                    "overall_validation": "PASSED_WITH_WARNINGS"
                }
            else:  # PLAN
                result = {
                    "qa_plan_created": True,
                    "test_strategy": "risk_based_testing",
                    "quality_gates_defined": True,
                    "estimated_effort": "2_weeks"
                }

            return Observation(
                result=result,
                success=True,
                metadata={"agent": "QA", "timestamp": datetime.now().isoformat()}
            )
        except Exception as e:
            return Observation(
                result={},
                success=False,
                error=str(e)
            )

class SchedulerAgent(ReActAgent):
    def __init__(self, llm_model: str = "gemini-1.5-flash"):
        super().__init__(
            name="Scheduler_Agent",
            description="Scheduler agent responsible for task scheduling, resource allocation, and timeline management"
        )
        self.llm = ChatGoogleGenerativeAI(model=llm_model, temperature=0.1)

    async def think(self, context: Dict[str, Any]) -> Thought:
        current_task = context.get("task", "")

        prompt = f"""
        As a Scheduler Agent, analyze the current scheduling situation:

        Current Task: {current_task}
        Context: {context}

        Consider:
        1. Resource allocation requirements
        2. Timeline constraints
        3. Task dependencies
        4. Optimization opportunities

        What scheduling approach should I take?
        """

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])

        return Thought(
            content=response.content,
            reasoning="Analyzing scheduling requirements and resource constraints",
            confidence=0.85,
            timestamp=datetime.now().isoformat()
        )

    async def act(self, thought: Thought, context: Dict[str, Any]) -> Action:
        if "optimize" in thought.content.lower():
            return Action(
                type=ActionType.ANALYZE,
                description="Optimize resource allocation and scheduling",
                parameters={
                    "optimization_goal": "minimize_time_maximize_quality",
                    "resources": ["developers", "qa_engineers", "devops"],
                    "constraints": {"budget": 100000, "deadline": "3_months"}
                },
                expected_outcome="Optimized schedule with resource allocation plan"
            )
        elif "plan" in thought.content.lower():
            return Action(
                type=ActionType.PLAN,
                description="Create detailed project schedule",
                parameters={
                    "scheduling_algorithm": "critical_path_method",
                    "buffer_percentage": 0.2,
                    "milestone_tracking": True
                },
                expected_outcome="Comprehensive project schedule with milestones"
            )
        else:
            return Action(
                type=ActionType.EXECUTE,
                description="Execute scheduling optimization",
                parameters={
                    "task_prioritization": "high_value_first",
                    "resource_balancing": True
                },
                expected_outcome="Executed schedule with task assignments"
            )

    async def observe(self, action: Action) -> Observation:
        try:
            if action.type == ActionType.ANALYZE:
                result = {
                    "optimized_schedule": {
                        "total_duration": "10_weeks",
                        "resource_utilization": 0.87,
                        "critical_path": ["requirements", "design", "implementation", "testing", "deployment"]
                    },
                    "resource_allocation": {
                        "developers": 3,
                        "qa_engineers": 2,
                        "devops": 1
                    },
                    "cost_estimate": 85000
                }
            elif action.type == ActionType.PLAN:
                result = {
                    "schedule_created": True,
                    "total_tasks": 45,
                    "milestones": 8,
                    "estimated_completion": "2024-06-15"
                }
            else:  # EXECUTE
                result = {
                    "schedule_executed": True,
                    "tasks_assigned": 45,
                    "resource_conflicts_resolved": 3,
                    "efficiency_improvement": 0.23
                }

            return Observation(
                result=result,
                success=True,
                metadata={"agent": "Scheduler", "timestamp": datetime.now().isoformat()}
            )
        except Exception as e:
            return Observation(
                result={},
                success=False,
                error=str(e)
            )

class ProjectManagerAgent(ReActAgent):
    def __init__(self, llm_model: str = "gemini-1.5-flash"):
        super().__init__(
            name="Project_Manager_Agent",
            description="Project Manager agent responsible for strategic oversight, stakeholder management, and project coordination"
        )
        self.llm = ChatGoogleGenerativeAI(model=llm_model, temperature=0.1)

    async def think(self, context: Dict[str, Any]) -> Thought:
        current_task = context.get("task", "")

        prompt = f"""
        As a Project Manager Agent, analyze the current project situation:

        Current Task: {current_task}
        Context: {context}

        Consider:
        1. Strategic alignment
        2. Stakeholder requirements
        3. Risk management
        4. Project coordination needs

        What project management approach should I take?
        """

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])

        return Thought(
            content=response.content,
            reasoning="Analyzing project management requirements and strategic alignment",
            confidence=0.9,
            timestamp=datetime.now().isoformat()
        )

    async def act(self, thought: Thought, context: Dict[str, Any]) -> Action:
        if "plan" in thought.content.lower():
            return Action(
                type=ActionType.PLAN,
                description="Create comprehensive project plan and strategy",
                parameters={
                    "project_methodology": "agile_with_waterfall_gates",
                    "stakeholder_management": True,
                    "risk_assessment": True,
                    "communication_plan": True
                },
                expected_outcome="Complete project plan with strategy and governance framework"
            )
        elif "coordinate" in thought.content.lower():
            return Action(
                type=ActionType.COMMUNICATE,
                description="Coordinate cross-functional teams and stakeholders",
                parameters={
                    "communication_channels": ["daily_standups", "weekly_reviews", "monthly_reports"],
                    "stakeholder_updates": True,
                    "conflict_resolution": True
                },
                expected_outcome="Coordinated team activities with clear communication"
            )
        else:
            return Action(
                type=ActionType.ANALYZE,
                description="Analyze project status and performance",
                parameters={
                    "performance_metrics": ["scope", "schedule", "budget", "quality"],
                    "risk_assessment": True,
                    "stakeholder_satisfaction": True
                },
                expected_outcome="Project status report with performance analysis"
            )

    async def observe(self, action: Action) -> Observation:
        try:
            if action.type == ActionType.PLAN:
                result = {
                    "project_plan_created": True,
                    "strategic_alignment": "HIGH",
                    "risk_mitigation_strategies": 8,
                    "stakeholder_buy_in": 0.95,
                    "governance_framework": "established"
                }
            elif action.type == ActionType.COMMUNICATE:
                result = {
                    "teams_coordinated": 5,
                    "stakeholder_meetings": 3,
                    "issues_resolved": 7,
                    "communication_effectiveness": 0.92
                }
            else:  # ANALYZE
                result = {
                    "project_health": "GREEN",
                    "scope_completion": 0.75,
                    "schedule_variance": 0.02,
                    "budget_utilization": 0.68,
                    "quality_score": 0.88,
                    "risk_level": "LOW"
                }

            return Observation(
                result=result,
                success=True,
                metadata={"agent": "ProjectManager", "timestamp": datetime.now().isoformat()}
            )
        except Exception as e:
            return Observation(
                result={},
                success=False,
                error=str(e)
            )

# ============================================================================
# Multi-Agent Coordination System
# ============================================================================

class WorkflowState(Enum):
    INITIALIZING = "initializing"
    PLANNING = "planning"
    EXECUTING = "executing"
    COORDINATING = "coordinating"
    MONITORING = "monitoring"
    COMPLETING = "completing"

class AgentRole(Enum):
    QA = "qa"
    SCHEDULER = "scheduler"
    PROJECT_MANAGER = "project_manager"

@dataclass
class CoordinationMessage:
    sender: str
    recipient: str
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 1

class MultiAgentState(TypedDict):
    messages: List[BaseMessage]
    current_task: str
    workflow_state: str
    agent_states: Dict[str, Dict[str, Any]]
    coordination_messages: List[CoordinationMessage]
    shared_context: Dict[str, Any]
    results: Dict[str, Any]
    next_agent: str
    iteration_count: int
    max_iterations: int

class LangGraphCoordinator:
    def __init__(self, llm_model: str = "gemini-1.5-flash"):
        self.llm = ChatGoogleGenerativeAI(model=llm_model, temperature=0.1)

        # Initialize agents
        self.qa_agent = QAAgent(llm_model=llm_model)
        self.scheduler_agent = SchedulerAgent(llm_model=llm_model)
        self.project_manager_agent = ProjectManagerAgent(llm_model=llm_model)

        self.agents = {
            AgentRole.QA.value: self.qa_agent,
            AgentRole.SCHEDULER.value: self.scheduler_agent,
            AgentRole.PROJECT_MANAGER.value: self.project_manager_agent
        }

        self.message_queue: List[CoordinationMessage] = []
        self.workflow_graph = self._build_workflow_graph()

    def _build_workflow_graph(self) -> StateGraph:
        workflow = StateGraph(MultiAgentState)

        # Add nodes
        workflow.add_node("coordinator", self._coordinate)
        workflow.add_node("qa_agent", self._execute_qa_agent)
        workflow.add_node("scheduler_agent", self._execute_scheduler_agent)
        workflow.add_node("project_manager_agent", self._execute_project_manager_agent)
        workflow.add_node("decision_maker", self._make_routing_decision)
        workflow.add_node("finalizer", self._finalize_workflow)

        # Set entry point
        workflow.set_entry_point("coordinator")

        # Add edges
        workflow.add_conditional_edges(
            "coordinator",
            self._route_to_agent,
            {
                "qa": "qa_agent",
                "scheduler": "scheduler_agent",
                "project_manager": "project_manager_agent",
                "decision": "decision_maker",
                "end": "finalizer"
            }
        )

        for agent_name in ["qa_agent", "scheduler_agent", "project_manager_agent"]:
            workflow.add_conditional_edges(
                agent_name,
                self._check_completion,
                {
                    "continue": "decision_maker",
                    "end": "finalizer"
                }
            )

        workflow.add_conditional_edges(
            "decision_maker",
            self._route_next_action,
            {
                "qa": "qa_agent",
                "scheduler": "scheduler_agent",
                "project_manager": "project_manager_agent",
                "coordinator": "coordinator",
                "end": "finalizer"
            }
        )

        workflow.add_edge("finalizer", END)
        return workflow.compile()

    async def _coordinate(self, state: MultiAgentState) -> MultiAgentState:
        """Main coordination logic"""
        logger.info("Starting coordination phase")

        coordination_plan = {
            "required_agents": ["project_manager", "scheduler", "qa"],
            "first_agent": "project_manager",
            "execution_strategy": "sequential_with_coordination"
        }

        state["shared_context"].update({
            "coordination_plan": coordination_plan,
            "coordination_timestamp": datetime.now().isoformat(),
            "active_agents": coordination_plan.get("required_agents", [])
        })

        state["workflow_state"] = WorkflowState.PLANNING.value
        state["next_agent"] = "project_manager"

        return state

    async def _execute_qa_agent(self, state: MultiAgentState) -> MultiAgentState:
        """Execute QA Agent"""
        logger.info("Executing QA Agent")

        qa_context = {
            "task": state["current_task"],
            "shared_context": state["shared_context"],
            "agent_states": state["agent_states"]
        }

        qa_result = await self.qa_agent.reason_act_cycle(qa_context)

        state["agent_states"]["qa"] = {
            "last_execution": datetime.now().isoformat(),
            "result": qa_result,
            "status": "completed" if qa_result.get("success") else "failed",
            "memory_summary": self.qa_agent.get_memory_summary()
        }

        state["results"]["qa_agent"] = qa_result
        state["iteration_count"] += 1
        return state

    async def _execute_scheduler_agent(self, state: MultiAgentState) -> MultiAgentState:
        """Execute Scheduler Agent"""
        logger.info("Executing Scheduler Agent")

        scheduler_context = {
            "task": state["current_task"],
            "shared_context": state["shared_context"],
            "agent_states": state["agent_states"],
            "qa_results": state["results"].get("qa_agent", {})
        }

        scheduler_result = await self.scheduler_agent.reason_act_cycle(scheduler_context)

        state["agent_states"]["scheduler"] = {
            "last_execution": datetime.now().isoformat(),
            "result": scheduler_result,
            "status": "completed" if scheduler_result.get("success") else "failed",
            "memory_summary": self.scheduler_agent.get_memory_summary()
        }

        state["results"]["scheduler_agent"] = scheduler_result
        state["iteration_count"] += 1
        return state

    async def _execute_project_manager_agent(self, state: MultiAgentState) -> MultiAgentState:
        """Execute Project Manager Agent"""
        logger.info("Executing Project Manager Agent")

        pm_context = {
            "task": state["current_task"],
            "shared_context": state["shared_context"],
            "agent_states": state["agent_states"],
            "qa_results": state["results"].get("qa_agent", {}),
            "scheduler_results": state["results"].get("scheduler_agent", {})
        }

        pm_result = await self.project_manager_agent.reason_act_cycle(pm_context)

        state["agent_states"]["project_manager"] = {
            "last_execution": datetime.now().isoformat(),
            "result": pm_result,
            "status": "completed" if pm_result.get("success") else "failed",
            "memory_summary": self.project_manager_agent.get_memory_summary()
        }

        state["results"]["project_manager_agent"] = pm_result
        state["iteration_count"] += 1
        return state

    async def _make_routing_decision(self, state: MultiAgentState) -> MultiAgentState:
        """Make routing decision"""
        logger.info("Making routing decision")

        # Simple decision logic
        completed_agents = [
            agent for agent, agent_state in state["agent_states"].items()
            if agent_state.get("status") == "completed"
        ]

        if len(completed_agents) >= 3 or state["iteration_count"] >= state["max_iterations"]:
            state["next_agent"] = "end"
        else:
            # Determine next agent
            if "project_manager" not in completed_agents:
                state["next_agent"] = "project_manager"
            elif "scheduler" not in completed_agents:
                state["next_agent"] = "scheduler"
            elif "qa" not in completed_agents:
                state["next_agent"] = "qa"
            else:
                state["next_agent"] = "end"

        return state

    async def _finalize_workflow(self, state: MultiAgentState) -> MultiAgentState:
        """Finalize workflow"""
        logger.info("Finalizing workflow")

        final_results = {
            "task": state["current_task"],
            "workflow_completed": True,
            "total_iterations": state["iteration_count"],
            "agent_results": state["results"],
            "success": all(result.get("success", False) for result in state["results"].values())
        }

        state["results"]["final_workflow_result"] = final_results
        state["workflow_state"] = WorkflowState.COMPLETING.value

        return state

    def _route_to_agent(self, state: MultiAgentState) -> str:
        """Route to appropriate agent"""
        next_agent = state.get("next_agent", "project_manager")

        if state["iteration_count"] >= state["max_iterations"]:
            return "end"

        return next_agent

    def _check_completion(self, state: MultiAgentState) -> str:
        """Check if workflow should continue"""
        if state["iteration_count"] >= state["max_iterations"]:
            return "end"

        return "continue"

    def _route_next_action(self, state: MultiAgentState) -> str:
        """Route next action"""
        next_agent = state.get("next_agent", "end")

        if state["iteration_count"] >= state["max_iterations"]:
            return "end"

        return next_agent

    async def execute_workflow(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the complete multi-agent workflow"""

        initial_state: MultiAgentState = {
            "messages": [HumanMessage(content=task)],
            "current_task": task,
            "workflow_state": WorkflowState.INITIALIZING.value,
            "agent_states": {},
            "coordination_messages": [],
            "shared_context": context or {},
            "results": {},
            "next_agent": "project_manager",
            "iteration_count": 0,
            "max_iterations": 10
        }

        try:
            final_state = await self.workflow_graph.ainvoke(initial_state)
            return final_state.get("results", {}).get("final_workflow_result", {
                "error": "Workflow completed but no final result generated",
                "success": False
            })
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            return {
                "error": str(e),
                "success": False,
                "partial_results": initial_state.get("results", {})
            }

# ============================================================================
# Main System Orchestrator
# ============================================================================

class CombinedMultiAgentSystem:
    """Combined Multi-Agent System with Qdrant integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._load_default_config()

        # Initialize components
        self.coordinator = LangGraphCoordinator(
            llm_model=self.config["llm"]["model"]
        )

        self.vector_store = QdrantVectorStore(
            host=self.config["qdrant"]["host"],
            port=self.config["qdrant"]["port"],
            api_key=self.config["qdrant"].get("api_key"),
            embedding_model=self.config["qdrant"]["embedding_model"]
        )

        self.is_running = False
        self.startup_time = None

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            "system": {
                "name": "Combined Multi-Agent System",
                "version": "1.0.0",
                "environment": os.getenv("ENVIRONMENT", "development")
            },
            "llm": {
                "model": "gemini-1.5-flash",
                "temperature": 0.1,
                "api_key": os.getenv("GOOGLE_API_KEY")
            },
            "qdrant": {
                "host": os.getenv("QDRANT_HOST", "localhost"),
                "port": 6333,
                "api_key": os.getenv("QDRANT_API_KEY"),
                "embedding_model": "all-MiniLM-L6-v2"
            }
        }

    async def initialize(self) -> bool:
        """Initialize the system"""
        logger.info("Initializing Combined Multi-Agent System...")

        try:
            # Initialize Qdrant collections (skip if Qdrant not available)
            try:
                collections_result = await self.vector_store.initialize_collections()
                vector_success = all(collections_result.values())
                if not vector_success:
                    logger.warning("Some Qdrant collections failed to initialize")
            except Exception as e:
                logger.warning(f"Qdrant not available, running without vector storage: {e}")

            self.is_running = True
            self.startup_time = datetime.now()

            logger.info("Combined Multi-Agent System initialized successfully!")
            return True

        except Exception as e:
            logger.error(f"System initialization failed: {str(e)}")
            return False

    async def execute_task(self,
                          task_description: str,
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a task using the multi-agent system"""

        if not self.is_running:
            return {
                "success": False,
                "error": "System not initialized",
                "timestamp": datetime.now().isoformat()
            }

        logger.info(f"Executing task: {task_description}")

        execution_context = context or {}
        execution_context.update({
            "vector_storage_available": True,
            "system_config": self.config
        })

        try:
            # Execute workflow through coordinator
            result = await self.coordinator.execute_workflow(
                task=task_description,
                context=execution_context
            )

            # Store execution results in vector database
            if result.get("success"):
                await self._store_execution_results(task_description, result, execution_context)

            logger.info(f"Task execution completed. Success: {result.get('success', False)}")
            return result

        except Exception as e:
            logger.error(f"Task execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "task": task_description
            }

    async def _store_execution_results(self,
                                     task: str,
                                     result: Dict[str, Any],
                                     context: Dict[str, Any]):
        """Store execution results in vector database"""
        try:
            content = f"""
            Task: {task}
            Result: {json.dumps(result, indent=2)}
            Success: {result.get('success', False)}
            Timestamp: {datetime.now().isoformat()}
            """

            metadata = DocumentMetadata(
                agent_name="orchestrator",
                document_type="execution_result",
                timestamp=datetime.now().isoformat(),
                task_id=context.get('task_id'),
                project_id=context.get('project_id'),
                tags=["orchestrator", "execution", "result"],
                priority=2
            )

            await self.vector_store.store_agent_knowledge(
                agent_name="orchestrator",
                content=content,
                metadata=metadata,
                collection_name="coordination_logs"
            )
        except Exception as e:
            logger.error(f"Failed to store execution results: {str(e)}")

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "system_info": {
                "name": self.config["system"]["name"],
                "version": self.config["system"]["version"],
                "environment": self.config["system"]["environment"],
                "startup_time": self.startup_time.isoformat() if self.startup_time else None,
                "is_running": self.is_running
            },
            "components": {
                "coordinator": "active" if self.coordinator else "inactive",
                "vector_store": "active" if self.vector_store else "inactive"
            },
            "agents": {
                "enabled": ["qa", "scheduler", "project_manager"],
                "total_agents": len(self.coordinator.agents) if self.coordinator else 0
            }
        }

    async def search_knowledge(self,
                             query: str,
                             collection_name: str = "agent_knowledge",
                             limit: int = 5) -> List[Dict[str, Any]]:
        """Search knowledge in vector database"""
        return await self.vector_store.search_similar_knowledge(
            query=query,
            collection_name=collection_name,
            limit=limit
        )

# ============================================================================
# Example Usage and Main Function
# ============================================================================

async def run_example():
    """Run an example demonstration"""
    system = CombinedMultiAgentSystem()

    # Initialize system
    if not await system.initialize():
        logger.error("Failed to initialize system")
        return

    # Example tasks
    example_tasks = [
        {
            "description": "Analyze project requirements and create development plan",
            "context": {
                "project_type": "web_application",
                "timeline": "3_months",
                "team_size": 5
            }
        },
        {
            "description": "Perform quality assurance review and testing strategy",
            "context": {
                "code_base": "python_flask_app",
                "test_coverage_target": 85,
                "quality_gates": ["unit_tests", "integration_tests", "security_scan"]
            }
        }
    ]

    # Execute example tasks
    for i, task_info in enumerate(example_tasks, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"EXECUTING EXAMPLE TASK {i}")
        logger.info(f"{'='*60}")

        result = await system.execute_task(
            task_description=task_info["description"],
            context=task_info["context"]
        )

        logger.info(f"Task {i} Result Summary:")
        logger.info(f"Success: {result.get('success', False)}")

        if result.get("success"):
            logger.info("✅ Task completed successfully!")
        else:
            logger.error(f"❌ Task failed: {result.get('error', 'Unknown error')}")

        # Wait between tasks
        await asyncio.sleep(1)

    # Get final system status
    logger.info(f"\n{'='*60}")
    logger.info("FINAL SYSTEM STATUS")
    logger.info(f"{'='*60}")

    status = await system.get_system_status()
    logger.info(f"System Status: {json.dumps(status, indent=2, default=str)}")

async def interactive_mode():
    """Run in interactive mode"""
    system = CombinedMultiAgentSystem()

    # Initialize system
    if not await system.initialize():
        logger.error("Failed to initialize system")
        return

    print("\n" + "="*60)
    print("COMBINED MULTI-AGENT SYSTEM - INTERACTIVE MODE")
    print("="*60)
    print("Enter tasks for the multi-agent system to execute.")
    print("Type 'status' to see system status.")
    print("Type 'search <query>' to search knowledge base.")
    print("Type 'exit' to quit.")
    print("="*60 + "\n")

    while True:
        try:
            user_input = input("\nEnter command: ").strip()

            if user_input.lower() == 'exit':
                break
            elif user_input.lower() == 'status':
                status = await system.get_system_status()
                print(json.dumps(status, indent=2, default=str))
                continue
            elif user_input.lower().startswith('search '):
                query = user_input[7:]  # Remove 'search '
                results = await system.search_knowledge(query)
                print(f"\nFound {len(results)} results:")
                for result in results:
                    print(f"- {result['content'][:100]}... (Score: {result['score']:.3f})")
                continue
            elif not user_input:
                continue

            print(f"\nExecuting: {user_input}")
            print("-" * 40)

            result = await system.execute_task(user_input)

            print(f"\nResult Summary:")
            print(f"Success: {result.get('success', False)}")

            if result.get('success'):
                print("✅ Task completed successfully!")

                # Show agent summaries
                agent_results = result.get('agent_results', {})
                if agent_results:
                    print("\nAgent Contributions:")
                    for agent, agent_result in agent_results.items():
                        if isinstance(agent_result, dict):
                            print(f"  • {agent.upper()}: {agent_result.get('success', 'Unknown status')}")
            else:
                print(f"❌ Task failed: {result.get('error', 'Unknown error')}")

        except KeyboardInterrupt:
            print("\n\nReceived interrupt signal...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

    print("\nShutting down...")
    print("Goodbye!")

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Combined Multi-Agent System")
    parser.add_argument(
        "--mode",
        choices=["interactive", "example"],
        default="interactive",
        help="Run mode (default: interactive)"
    )

    args = parser.parse_args()

    try:
        if args.mode == "interactive":
            asyncio.run(interactive_mode())
        elif args.mode == "example":
            asyncio.run(run_example())
        else:
            print(f"Unknown mode: {args.mode}")
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        logger.error(f"System error: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())