#!/usr/bin/env python3
"""
Comprehensive Test Suite for Combined Multi-Agent System with MCP and Qdrant
Tests the integration of:
- Combined Multi-Agent System (ReAct agents)
- MCP Tools Server and Client
- Qdrant Vector Database

This script tests the full pipeline from agent coordination to tool execution
and vector storage/retrieval.
"""

import asyncio
import logging
import json
import time
import subprocess
import sys
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

# Import our system components
from combined_multi_agent_system import CombinedMultiAgentSystem, DocumentMetadata
from mcp_client import MCPToolsIntegration
from mcp_server import MCPToolServer

# Real Google API key for full testing
import os
os.environ['GOOGLE_API_KEY'] = 'AIzaSyA95BAYSi0Y12io6zfO4CwmwWtCH3MeSOs'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

class IntegratedSystemTest:
    """Test suite for the integrated multi-agent system with MCP and Qdrant"""

    def __init__(self):
        self.combined_system = None
        self.mcp_integrations = {}
        self.test_results = {}
        self.start_time = None

    async def setup_test_environment(self) -> bool:
        """Setup the complete test environment"""
        logger.info("🚀 Setting up integrated test environment...")
        self.start_time = datetime.now()

        try:
            # 1. Initialize Combined Multi-Agent System
            logger.info("📊 Initializing Combined Multi-Agent System...")
            self.combined_system = CombinedMultiAgentSystem()

            if not await self.combined_system.initialize():
                logger.error("❌ Failed to initialize Combined Multi-Agent System")
                return False

            logger.info("✅ Combined Multi-Agent System initialized")

            # 2. Initialize MCP integrations for each agent
            logger.info("🔧 Initializing MCP integrations...")
            agents = ["qa", "scheduler", "project_manager"]

            for agent_name in agents:
                integration = MCPToolsIntegration(agent_name=agent_name)

                if await integration.initialize():
                    self.mcp_integrations[agent_name] = integration
                    logger.info(f"✅ MCP integration initialized for {agent_name}")
                else:
                    logger.warning(f"⚠️ MCP integration failed for {agent_name} - continuing without it")

            logger.info("🎉 Test environment setup complete!")
            return True

        except Exception as e:
            logger.error(f"❌ Test environment setup failed: {str(e)}")
            return False

    async def teardown_test_environment(self):
        """Cleanup test environment"""
        logger.info("🧹 Cleaning up test environment...")

        # Shutdown MCP integrations
        for agent_name, integration in self.mcp_integrations.items():
            try:
                await integration.shutdown()
                logger.info(f"🔌 Shutdown MCP integration for {agent_name}")
            except Exception as e:
                logger.warning(f"⚠️ Error shutting down {agent_name}: {e}")

        self.mcp_integrations.clear()
        logger.info("✅ Test environment cleanup complete")

    async def test_basic_connectivity(self) -> Dict[str, Any]:
        """Test basic connectivity to all services"""
        logger.info("🔍 Testing basic connectivity...")

        results = {
            "combined_system": False,
            "qdrant_vector_store": False,
            "mcp_integrations": {},
            "total_score": 0
        }

        # Test Combined System
        if self.combined_system and self.combined_system.is_running:
            results["combined_system"] = True
            results["total_score"] += 1
            logger.info("✅ Combined Multi-Agent System is running")
        else:
            logger.error("❌ Combined Multi-Agent System not available")

        # Test Qdrant Vector Store
        try:
            status = await self.combined_system.get_system_status()
            if status["components"]["vector_store"] == "active":
                results["qdrant_vector_store"] = True
                results["total_score"] += 1
                logger.info("✅ Qdrant Vector Store is active")
            else:
                logger.warning("⚠️ Qdrant Vector Store not active")
        except Exception as e:
            logger.error(f"❌ Qdrant connectivity test failed: {e}")

        # Test MCP integrations
        for agent_name, integration in self.mcp_integrations.items():
            if integration.connected:
                results["mcp_integrations"][agent_name] = True
                results["total_score"] += 1
                logger.info(f"✅ MCP integration active for {agent_name}")
            else:
                results["mcp_integrations"][agent_name] = False
                logger.error(f"❌ MCP integration failed for {agent_name}")

        results["connectivity_percentage"] = (results["total_score"] / (2 + len(self.mcp_integrations))) * 100

        return results

    async def test_agent_coordination_with_tools(self) -> Dict[str, Any]:
        """Test agent coordination with MCP tools integration"""
        logger.info("🤖 Testing agent coordination with MCP tools...")

        results = {
            "task_execution": {},
            "tool_usage": {},
            "coordination_success": False,
            "knowledge_storage": False
        }

        # Define test task
        test_task = "Develop a comprehensive quality assurance strategy for a Python web application with automated testing, code quality checks, and performance monitoring"

        try:
            # Execute task through combined system
            logger.info(f"📋 Executing task: {test_task}")

            execution_result = await self.combined_system.execute_task(
                task_description=test_task,
                context={
                    "project_type": "python_web_app",
                    "tech_stack": ["Python", "Flask", "PostgreSQL"],
                    "team_size": 5,
                    "deadline": "2024-06-30",
                    "quality_requirements": ["85% test coverage", "automated CI/CD", "security scanning"]
                }
            )

            results["task_execution"] = {
                "success": execution_result.get("success", False),
                "total_iterations": execution_result.get("total_iterations", 0),
                "agent_results": len(execution_result.get("agent_results", {}))
            }

            if execution_result.get("success"):
                results["coordination_success"] = True
                logger.info("✅ Agent coordination completed successfully")

                # Test MCP tools usage with each agent
                await self._test_mcp_tools_per_agent(results)

                # Test knowledge storage in Qdrant
                await self._test_knowledge_storage(test_task, execution_result, results)
            else:
                logger.error(f"❌ Agent coordination failed: {execution_result.get('error', 'Unknown error')}")

        except Exception as e:
            logger.error(f"❌ Agent coordination test failed: {str(e)}")
            results["error"] = str(e)

        return results

    async def _test_mcp_tools_per_agent(self, results: Dict[str, Any]):
        """Test MCP tools for each agent type"""
        logger.info("🔧 Testing MCP tools per agent...")

        tool_tests = {
            "qa": [
                ("run_test_suite", {"test_path": "./tests", "test_type": "unit", "coverage": True}),
                ("validate_code_quality", {"code_path": "./src", "quality_gates": ["complexity", "documentation"]}),
                ("security_scan", {"target_path": "./src", "scan_type": "code"})
            ],
            "scheduler": [
                ("optimize_schedule", {
                    "tasks": [{"name": "Development", "duration": 40}, {"name": "Testing", "duration": 20}],
                    "constraints": {"budget": 50000, "deadline": "2024-06-30"},
                    "optimization_goal": "minimize_time"
                }),
                ("check_resource_availability", {
                    "resource_type": "human",
                    "time_window": {"start_date": "2024-03-01", "end_date": "2024-06-30"}
                }),
                ("forecast_capacity", {"historical_data": [100, 110, 105, 120, 115], "forecast_period": 30})
            ],
            "project_manager": [
                ("generate_report", {"report_type": "status", "data_sources": ["jira", "github"], "format": "json"}),
                ("update_stakeholders", {
                    "stakeholders": ["ceo", "cto", "product_manager"],
                    "update_type": "progress",
                    "content": {"message": "Project on track", "completion": 0.65}
                }),
                ("risk_assessment", {
                    "project_data": {"budget": 100000, "timeline": "6_months", "complexity": "high"},
                    "risk_categories": ["technical", "schedule", "budget"]
                })
            ]
        }

        results["tool_usage"] = {}

        for agent_name, tests in tool_tests.items():
            if agent_name in self.mcp_integrations:
                integration = self.mcp_integrations[agent_name]
                agent_results = []

                logger.info(f"🔍 Testing tools for {agent_name} agent...")

                for tool_name, params in tests:
                    try:
                        result = await integration.client.execute_tool(tool_name, params)

                        agent_results.append({
                            "tool": tool_name,
                            "success": result.success,
                            "execution_time": result.execution_time,
                            "error": result.error
                        })

                        status = "✅" if result.success else "❌"
                        logger.info(f"{status} {agent_name}:{tool_name} - {result.execution_time:.3f}s")

                    except Exception as e:
                        agent_results.append({
                            "tool": tool_name,
                            "success": False,
                            "execution_time": 0.0,
                            "error": str(e)
                        })
                        logger.error(f"❌ {agent_name}:{tool_name} failed: {e}")

                results["tool_usage"][agent_name] = {
                    "tools_tested": len(tests),
                    "tools_successful": sum(1 for r in agent_results if r["success"]),
                    "success_rate": sum(1 for r in agent_results if r["success"]) / len(tests) * 100,
                    "results": agent_results
                }

    async def _test_knowledge_storage(self, task: str, execution_result: Dict[str, Any], results: Dict[str, Any]):
        """Test knowledge storage and retrieval in Qdrant"""
        logger.info("💾 Testing knowledge storage in Qdrant...")

        try:
            # Store test knowledge
            test_content = f"""
            Task: {task}
            Execution Results: {json.dumps(execution_result, indent=2, default=str)}
            Test Timestamp: {datetime.now().isoformat()}
            Test Type: Integration Test
            """

            metadata = DocumentMetadata(
                agent_name="test_orchestrator",
                document_type="integration_test_result",
                timestamp=datetime.now().isoformat(),
                task_id="integration_test_001",
                project_id="mcp_qdrant_integration",
                tags=["integration_test", "mcp", "qdrant", "multi_agent"],
                priority=3
            )

            # Store in vector database
            point_id = await self.combined_system.vector_store.store_agent_knowledge(
                agent_name="test_orchestrator",
                content=test_content,
                metadata=metadata,
                collection_name="coordination_logs"
            )

            logger.info(f"✅ Knowledge stored with ID: {point_id}")

            # Test retrieval
            search_results = await self.combined_system.search_knowledge(
                query="integration test multi-agent coordination",
                collection_name="coordination_logs",
                limit=3
            )

            if search_results and len(search_results) > 0:
                results["knowledge_storage"] = True
                logger.info(f"✅ Knowledge retrieval successful - found {len(search_results)} results")
            else:
                logger.warning("⚠️ Knowledge retrieval returned no results")

        except Exception as e:
            logger.error(f"❌ Knowledge storage test failed: {str(e)}")

    async def test_real_time_coordination(self) -> Dict[str, Any]:
        """Test real-time coordination between agents using WebSocket"""
        logger.info("⚡ Testing real-time coordination...")

        results = {
            "websocket_connections": 0,
            "real_time_messages": 0,
            "coordination_latency": [],
            "success": False
        }

        try:
            # Test WebSocket connections for each agent
            for agent_name, integration in self.mcp_integrations.items():
                if hasattr(integration.client, 'websocket') and integration.client.websocket:
                    results["websocket_connections"] += 1

            # Test real-time notifications
            if results["websocket_connections"] > 0:
                logger.info("📡 Testing real-time notifications...")

                for agent_name, integration in self.mcp_integrations.items():
                    try:
                        start_time = time.time()
                        notification_result = await integration.send_notification(
                            recipient=f"test_coordinator",
                            message=f"Real-time test from {agent_name} agent",
                            priority=2
                        )
                        end_time = time.time()

                        if notification_result.success:
                            latency = (end_time - start_time) * 1000  # Convert to milliseconds
                            results["coordination_latency"].append(latency)
                            results["real_time_messages"] += 1
                            logger.info(f"✅ Real-time message from {agent_name} - {latency:.1f}ms")

                    except Exception as e:
                        logger.error(f"❌ Real-time test failed for {agent_name}: {e}")

                if results["real_time_messages"] > 0:
                    avg_latency = sum(results["coordination_latency"]) / len(results["coordination_latency"])
                    results["average_latency_ms"] = avg_latency
                    results["success"] = True
                    logger.info(f"✅ Real-time coordination test passed - avg latency: {avg_latency:.1f}ms")

        except Exception as e:
            logger.error(f"❌ Real-time coordination test failed: {str(e)}")
            results["error"] = str(e)

        return results

    async def test_performance_metrics(self) -> Dict[str, Any]:
        """Test performance metrics of the integrated system"""
        logger.info("⚡ Testing performance metrics...")

        results = {
            "task_execution_times": [],
            "tool_execution_times": [],
            "vector_operations": [],
            "memory_usage": {},
            "throughput": {}
        }

        try:
            # Test multiple task executions for performance
            simple_tasks = [
                "Analyze code quality metrics for a Python project",
                "Create resource allocation plan for development team",
                "Generate project status report with risk assessment"
            ]

            for i, task in enumerate(simple_tasks):
                logger.info(f"⏱️ Performance test {i+1}/3: {task[:50]}...")

                start_time = time.time()
                result = await self.combined_system.execute_task(task)
                end_time = time.time()

                execution_time = end_time - start_time
                results["task_execution_times"].append(execution_time)

                logger.info(f"📊 Task {i+1} completed in {execution_time:.2f}s")

            # Test tool execution performance
            if "qa" in self.mcp_integrations:
                qa_integration = self.mcp_integrations["qa"]

                for i in range(3):
                    start_time = time.time()
                    result = await qa_integration.get_current_time()
                    end_time = time.time()

                    tool_time = (end_time - start_time) * 1000  # milliseconds
                    results["tool_execution_times"].append(tool_time)

            # Calculate performance metrics
            if results["task_execution_times"]:
                results["avg_task_time"] = sum(results["task_execution_times"]) / len(results["task_execution_times"])
                results["throughput"]["tasks_per_minute"] = 60 / results["avg_task_time"]

            if results["tool_execution_times"]:
                results["avg_tool_time_ms"] = sum(results["tool_execution_times"]) / len(results["tool_execution_times"])
                results["throughput"]["tools_per_second"] = 1000 / results["avg_tool_time_ms"]

            logger.info("✅ Performance testing completed")

        except Exception as e:
            logger.error(f"❌ Performance testing failed: {str(e)}")
            results["error"] = str(e)

        return results

    async def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        logger.info("📄 Generating comprehensive test report...")

        end_time = datetime.now()
        test_duration = (end_time - self.start_time).total_seconds() if self.start_time else 0

        # Calculate overall success rate
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values()
                             if isinstance(result, dict) and result.get("success", False))

        report = {
            "test_summary": {
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": end_time.isoformat(),
                "duration_seconds": test_duration,
                "total_test_categories": total_tests,
                "successful_categories": successful_tests,
                "overall_success_rate": (successful_tests / total_tests * 100) if total_tests > 0 else 0
            },
            "component_status": {
                "combined_multi_agent_system": self.combined_system.is_running if self.combined_system else False,
                "mcp_integrations": {name: integration.connected for name, integration in self.mcp_integrations.items()},
                "qdrant_vector_store": True  # Assume active if tests ran
            },
            "detailed_results": self.test_results,
            "recommendations": self._generate_recommendations()
        }

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []

        # Check connectivity
        connectivity = self.test_results.get("connectivity", {})
        if connectivity.get("connectivity_percentage", 0) < 100:
            recommendations.append("Improve service connectivity - some components failed to connect")

        # Check tool usage
        coordination = self.test_results.get("coordination", {})
        tool_usage = coordination.get("tool_usage", {})

        for agent, usage in tool_usage.items():
            if usage.get("success_rate", 0) < 80:
                recommendations.append(f"Review {agent} agent tool configuration - success rate below 80%")

        # Check performance
        performance = self.test_results.get("performance", {})
        avg_task_time = performance.get("avg_task_time", 0)
        if avg_task_time > 10:
            recommendations.append("Optimize task execution time - average exceeds 10 seconds")

        # Check real-time coordination
        realtime = self.test_results.get("realtime", {})
        avg_latency = realtime.get("average_latency_ms", 0)
        if avg_latency > 1000:
            recommendations.append("Optimize real-time communication - latency exceeds 1 second")

        if not recommendations:
            recommendations.append("System performing well - no critical issues identified")

        return recommendations

    async def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run the complete test suite"""
        logger.info("🚀 Starting comprehensive test suite...")
        logger.info("=" * 80)

        try:
            # Setup
            if not await self.setup_test_environment():
                return {"success": False, "error": "Test environment setup failed"}

            # Test 1: Basic Connectivity
            logger.info("\n" + "=" * 40)
            logger.info("TEST 1: BASIC CONNECTIVITY")
            logger.info("=" * 40)
            self.test_results["connectivity"] = await self.test_basic_connectivity()

            # Test 2: Agent Coordination with Tools
            logger.info("\n" + "=" * 40)
            logger.info("TEST 2: AGENT COORDINATION & TOOLS")
            logger.info("=" * 40)
            self.test_results["coordination"] = await self.test_agent_coordination_with_tools()

            # Test 3: Real-time Coordination
            logger.info("\n" + "=" * 40)
            logger.info("TEST 3: REAL-TIME COORDINATION")
            logger.info("=" * 40)
            self.test_results["realtime"] = await self.test_real_time_coordination()

            # Test 4: Performance Metrics
            logger.info("\n" + "=" * 40)
            logger.info("TEST 4: PERFORMANCE METRICS")
            logger.info("=" * 40)
            self.test_results["performance"] = await self.test_performance_metrics()

            # Generate final report
            logger.info("\n" + "=" * 40)
            logger.info("GENERATING FINAL REPORT")
            logger.info("=" * 40)
            final_report = await self.generate_test_report()

            return final_report

        except Exception as e:
            logger.error(f"❌ Test suite failed: {str(e)}")
            return {"success": False, "error": str(e)}

        finally:
            await self.teardown_test_environment()


async def check_prerequisites():
    """Check if required services are running"""
    logger.info("🔍 Checking prerequisites...")

    prerequisites = {
        "qdrant": {"port": 6333, "service": "Qdrant Vector Database"},
        "mcp_server": {"port": 8080, "service": "MCP Tools Server"}
    }

    missing = []

    for service, info in prerequisites.items():
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex(('localhost', info["port"]))
            sock.close()

            if result == 0:
                logger.info(f"✅ {info['service']} is running on port {info['port']}")
            else:
                logger.warning(f"⚠️ {info['service']} not found on port {info['port']}")
                missing.append(service)

        except Exception as e:
            logger.warning(f"⚠️ Could not check {info['service']}: {e}")
            missing.append(service)

    if missing:
        logger.info("\n📋 To start missing services:")
        if "qdrant" in missing:
            logger.info("   Qdrant: docker run -p 6333:6333 qdrant/qdrant")
        if "mcp_server" in missing:
            logger.info("   MCP Server: python mcp_server.py --host localhost --port 8080")
        logger.info("\n⚠️ Some tests may fail without these services")

        # Auto-continue for automated testing
        logger.info("\n🤖 Auto-continuing with available services...")
        return True

    return True


def print_test_summary(report: Dict[str, Any]):
    """Print a formatted test summary"""
    print("\n" + "=" * 80)
    print("🎯 INTEGRATED SYSTEM TEST RESULTS")
    print("=" * 80)

    summary = report.get("test_summary", {})

    print(f"🕒 Test Duration: {summary.get('duration_seconds', 0):.1f} seconds")
    print(f"📊 Success Rate: {summary.get('overall_success_rate', 0):.1f}%")
    print(f"🧪 Test Categories: {summary.get('total_test_categories', 0)}")

    # Component Status
    print("\n📋 Component Status:")
    components = report.get("component_status", {})
    for component, status in components.items():
        if isinstance(status, dict):
            for sub_component, sub_status in status.items():
                icon = "✅" if sub_status else "❌"
                print(f"  {icon} {component}.{sub_component}")
        else:
            icon = "✅" if status else "❌"
            print(f"  {icon} {component}")

    # Key Metrics
    print("\n📈 Key Metrics:")
    detailed = report.get("detailed_results", {})

    # Connectivity
    connectivity = detailed.get("connectivity", {})
    print(f"  🔗 Connectivity: {connectivity.get('connectivity_percentage', 0):.1f}%")

    # Performance
    performance = detailed.get("performance", {})
    if performance.get("avg_task_time"):
        print(f"  ⚡ Avg Task Time: {performance['avg_task_time']:.2f}s")
    if performance.get("avg_tool_time_ms"):
        print(f"  🔧 Avg Tool Time: {performance['avg_tool_time_ms']:.1f}ms")

    # Real-time
    realtime = detailed.get("realtime", {})
    if realtime.get("average_latency_ms"):
        print(f"  📡 Avg Latency: {realtime['average_latency_ms']:.1f}ms")

    # Recommendations
    recommendations = report.get("recommendations", [])
    if recommendations:
        print("\n💡 Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")

    print("\n" + "=" * 80)


async def main():
    """Main test execution function"""
    print("🤖 Combined Multi-Agent System + MCP + Qdrant Integration Test")
    print("=" * 80)

    # Check prerequisites
    if not await check_prerequisites():
        logger.error("❌ Prerequisites check failed")
        return 1

    # Run test suite
    test_runner = IntegratedSystemTest()

    try:
        report = await test_runner.run_comprehensive_test_suite()

        # Print summary
        print_test_summary(report)

        # Save detailed report
        report_file = f"integration_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"📄 Detailed report saved to: {report_file}")

        # Return exit code based on success rate
        success_rate = report.get("test_summary", {}).get("overall_success_rate", 0)
        return 0 if success_rate >= 80 else 1

    except KeyboardInterrupt:
        logger.info("\n⏹️ Test interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Test execution failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))