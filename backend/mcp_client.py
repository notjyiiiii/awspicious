#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Client
A client for connecting to MCP Tools Server with full async support

Features:
- HTTP REST API client for tool execution
- WebSocket client for real-time communication
- Agent-specific tool access management
- Built-in convenience methods for all 13+ tools
- Request history and monitoring
- Async/await support throughout
"""

import asyncio
import aiohttp
import json
import logging
import websockets
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Core MCP Client Types
# ============================================================================

@dataclass
class MCPToolResult:
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: str = ""
    tool_name: str = ""
    request_id: str = ""

# ============================================================================
# Base MCP Client
# ============================================================================

class MCPClient:
    """Client for connecting to MCP Tools Server"""

    def __init__(self,
                 server_url: str = "http://localhost:8080",
                 websocket_url: str = "ws://localhost:8080/ws",
                 agent_name: str = "unknown"):
        self.server_url = server_url.rstrip('/')
        self.websocket_url = websocket_url
        self.agent_name = agent_name
        self.websocket = None
        self.session = None
        self.available_tools = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.connected = False

    async def connect(self) -> bool:
        """Connect to the MCP server"""
        try:
            self.session = aiohttp.ClientSession()

            # Test connection
            async with self.session.get(f"{self.server_url}/") as response:
                if response.status == 200:
                    server_info = await response.json()
                    logger.info(f"✅ Connected to MCP server: {self.server_url}")
                    logger.info(f"📊 Server info: {server_info.get('message', 'MCP Server')}")

                    # Load available tools
                    await self._load_available_tools()
                    self.connected = True
                    return True
                else:
                    logger.error(f"❌ Failed to connect to MCP server: HTTP {response.status}")
                    return False

        except Exception as e:
            logger.error(f"❌ Connection failed: {str(e)}")
            return False

    async def disconnect(self):
        """Disconnect from the MCP server"""
        if self.websocket:
            await self.websocket.close()
            logger.info("🔌 WebSocket disconnected")

        if self.session:
            await self.session.close()
            logger.info("🔌 HTTP session closed")

        self.connected = False
        logger.info("👋 Disconnected from MCP server")

    async def _load_available_tools(self):
        """Load list of available tools from server"""
        try:
            async with self.session.get(f"{self.server_url}/tools") as response:
                if response.status == 200:
                    data = await response.json()
                    self.available_tools = data.get("tools", {})
                    total_tools = data.get("total_tools", len(self.available_tools))

                    # Get agent-specific tools
                    agent_tools = await self._get_agent_tools()

                    logger.info(f"🔧 Loaded {total_tools} total tools")
                    logger.info(f"👤 Agent '{self.agent_name}' has access to {len(agent_tools)} tools")
                else:
                    logger.error(f"❌ Failed to load tools: HTTP {response.status}")

        except Exception as e:
            logger.error(f"❌ Failed to load tools: {str(e)}")

    async def _get_agent_tools(self) -> Dict[str, Any]:
        """Get tools available for this specific agent"""
        try:
            async with self.session.get(f"{self.server_url}/agents/{self.agent_name}/tools") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("available_tools", {})
                else:
                    # Fallback to filtering from all tools
                    return self._filter_agent_tools()
        except Exception as e:
            logger.error(f"❌ Failed to get agent tools: {str(e)}")
            return {}

    def _filter_agent_tools(self) -> Dict[str, Any]:
        """Filter tools available for this agent"""
        agent_tools = {}
        for tool_name, tool_info in self.available_tools.items():
            agent_access = tool_info.get("agent_access", [])
            if self.agent_name in agent_access or "all" in agent_access:
                agent_tools[tool_name] = tool_info
        return agent_tools

    async def execute_tool(self,
                          tool_name: str,
                          parameters: Dict[str, Any],
                          request_id: Optional[str] = None) -> MCPToolResult:
        """Execute a tool on the MCP server"""

        if not self.session or not self.connected:
            return MCPToolResult(
                success=False,
                error="Not connected to MCP server",
                tool_name=tool_name
            )

        try:
            request_data = {
                "tool_name": tool_name,
                "parameters": parameters,
                "agent_name": self.agent_name,
                "request_id": request_id
            }

            logger.info(f"🔧 Executing tool: {tool_name}")

            async with self.session.post(
                f"{self.server_url}/execute",
                json=request_data
            ) as response:

                if response.status == 200:
                    data = await response.json()
                    result = MCPToolResult(
                        success=data.get("success", False),
                        result=data.get("result"),
                        error=data.get("error"),
                        execution_time=data.get("execution_time", 0.0),
                        timestamp=data.get("timestamp", ""),
                        tool_name=tool_name,
                        request_id=data.get("request_id", "")
                    )

                    if result.success:
                        logger.info(f"✅ Tool '{tool_name}' executed successfully ({result.execution_time:.3f}s)")
                    else:
                        logger.error(f"❌ Tool '{tool_name}' failed: {result.error}")

                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"❌ HTTP {response.status}: {error_text}")
                    return MCPToolResult(
                        success=False,
                        error=f"HTTP {response.status}: {error_text}",
                        tool_name=tool_name
                    )

        except Exception as e:
            logger.error(f"❌ Tool execution failed: {str(e)}")
            return MCPToolResult(
                success=False,
                error=str(e),
                tool_name=tool_name
            )

    async def connect_websocket(self) -> bool:
        """Connect to WebSocket for real-time communication"""
        try:
            self.websocket = await websockets.connect(self.websocket_url)
            logger.info("🔗 WebSocket connection established")

            # Start message handler
            asyncio.create_task(self._handle_websocket_messages())
            return True

        except Exception as e:
            logger.error(f"❌ WebSocket connection failed: {str(e)}")
            return False

    async def _handle_websocket_messages(self):
        """Handle incoming WebSocket messages"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                message_type = data.get("type")

                logger.info(f"📨 Received WebSocket message: {message_type}")

                if message_type in self.message_handlers:
                    handler = self.message_handlers[message_type]
                    try:
                        await handler(data.get("data", {}))
                    except Exception as e:
                        logger.error(f"❌ Message handler error: {str(e)}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("🔌 WebSocket connection closed")
        except Exception as e:
            logger.error(f"❌ WebSocket error: {str(e)}")

    def register_message_handler(self, message_type: str, handler: Callable):
        """Register a handler for specific message types"""
        self.message_handlers[message_type] = handler
        logger.info(f"📝 Registered handler for message type: {message_type}")

    async def send_websocket_message(self, message_type: str, data: Any):
        """Send a message via WebSocket"""
        if not self.websocket:
            logger.error("❌ WebSocket not connected")
            return False

        try:
            message = {
                "type": message_type,
                "data": data
            }
            await self.websocket.send(json.dumps(message))
            logger.info(f"📤 Sent WebSocket message: {message_type}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to send WebSocket message: {str(e)}")
            return False

    async def execute_tool_async(self,
                               tool_name: str,
                               parameters: Dict[str, Any],
                               callback: Optional[Callable] = None) -> str:
        """Execute tool asynchronously via WebSocket"""

        if not self.websocket:
            raise Exception("WebSocket not connected")

        request_id = f"{self.agent_name}_{datetime.now().timestamp()}"

        # Register callback if provided
        if callback:
            async def response_handler(data):
                if data.get("request_id") == request_id:
                    await callback(data)

            self.register_message_handler("tool_response", response_handler)

        # Send tool request
        await self.send_websocket_message("tool_request", {
            "tool_name": tool_name,
            "parameters": parameters,
            "agent_name": self.agent_name,
            "request_id": request_id
        })

        return request_id

    def get_available_tools(self) -> Dict[str, Any]:
        """Get list of available tools"""
        return self.available_tools.copy()

    def is_tool_available(self, tool_name: str) -> bool:
        """Check if a tool is available for this agent"""
        if tool_name not in self.available_tools:
            return False

        tool_info = self.available_tools[tool_name]
        agent_access = tool_info.get("agent_access", [])

        return self.agent_name in agent_access or "all" in agent_access

    async def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific tool"""
        if not self.session:
            return None

        try:
            async with self.session.get(f"{self.server_url}/tools/{tool_name}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"❌ Failed to get tool info: HTTP {response.status}")
                    return None

        except Exception as e:
            logger.error(f"❌ Failed to get tool info: {str(e)}")
            return None

    async def get_execution_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent tool execution history"""
        if not self.session:
            return []

        try:
            async with self.session.get(f"{self.server_url}/history?limit={limit}") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("history", [])
                else:
                    logger.error(f"❌ Failed to get history: HTTP {response.status}")
                    return []

        except Exception as e:
            logger.error(f"❌ Failed to get history: {str(e)}")
            return []

# ============================================================================
# High-Level MCP Tools Integration
# ============================================================================

class MCPToolsIntegration:
    """Integration layer for agents to use MCP tools"""

    def __init__(self, agent_name: str, server_url: str = "http://localhost:8080"):
        self.agent_name = agent_name
        self.client = MCPClient(server_url=server_url, agent_name=agent_name)
        self.connected = False

    async def initialize(self) -> bool:
        """Initialize the MCP tools integration"""
        logger.info(f"🚀 Initializing MCP integration for agent: {self.agent_name}")

        self.connected = await self.client.connect()

        if self.connected:
            # Set up WebSocket connection for real-time features
            websocket_connected = await self.client.connect_websocket()

            if websocket_connected:
                # Register default message handlers
                self.client.register_message_handler("notification", self._handle_notification)
                self.client.register_message_handler("tool_execution", self._handle_tool_execution)

            logger.info(f"✅ MCP integration initialized successfully")
        else:
            logger.error(f"❌ Failed to initialize MCP integration")

        return self.connected

    async def shutdown(self):
        """Shutdown the integration"""
        logger.info(f"🔌 Shutting down MCP integration for agent: {self.agent_name}")
        await self.client.disconnect()
        self.connected = False

    async def _handle_notification(self, data: Dict[str, Any]):
        """Handle incoming notifications"""
        message = data.get('message', '')
        priority = data.get('priority', 1)
        logger.info(f"📢 Notification (Priority {priority}): {message}")

    async def _handle_tool_execution(self, data: Dict[str, Any]):
        """Handle tool execution notifications"""
        tool_name = data.get('tool_name', '')
        success = data.get('success', False)
        execution_time = data.get('execution_time', 0.0)
        status = "✅ Success" if success else "❌ Failed"
        logger.info(f"🔧 Tool executed: {tool_name} - {status} ({execution_time:.3f}s)")

    # ========================================================================
    # General Purpose Tools
    # ========================================================================

    async def get_current_time(self) -> MCPToolResult:
        """Get current time in multiple formats"""
        return await self.client.execute_tool("get_current_time", {})

    async def calculate_metrics(self, data: List[float], metrics: List[str]) -> MCPToolResult:
        """Calculate statistical metrics from data"""
        return await self.client.execute_tool("calculate_metrics", {
            "data": data,
            "metrics": metrics
        })

    async def send_notification(self, recipient: str, message: str, priority: int = 1) -> MCPToolResult:
        """Send notification to other agents or systems"""
        return await self.client.execute_tool("send_notification", {
            "recipient": recipient,
            "message": message,
            "priority": priority
        })

    async def read_file(self, file_path: str, encoding: str = "utf-8") -> MCPToolResult:
        """Read file contents from filesystem"""
        return await self.client.execute_tool("read_file", {
            "file_path": file_path,
            "encoding": encoding
        })

    async def write_file(self, file_path: str, content: str, encoding: str = "utf-8") -> MCPToolResult:
        """Write content to file"""
        return await self.client.execute_tool("write_file", {
            "file_path": file_path,
            "content": content,
            "encoding": encoding
        })

    async def execute_command(self, command: str, timeout: int = 30) -> MCPToolResult:
        """Execute system command with timeout"""
        return await self.client.execute_tool("execute_command", {
            "command": command,
            "timeout": timeout
        })

    async def analyze_data(self, data: List[Any], analysis_type: str, options: Dict[str, Any] = None) -> MCPToolResult:
        """Perform data analysis"""
        return await self.client.execute_tool("analyze_data", {
            "data": data,
            "analysis_type": analysis_type,
            "options": options or {}
        })

    # ========================================================================
    # QA Agent Specific Tools
    # ========================================================================

    async def run_test_suite(self, test_path: str, test_type: str, coverage: bool = False) -> MCPToolResult:
        """Run comprehensive test suite (QA Agent only)"""
        if not self.client.is_tool_available("run_test_suite"):
            return MCPToolResult(success=False, error="Tool not available for this agent", tool_name="run_test_suite")

        return await self.client.execute_tool("run_test_suite", {
            "test_path": test_path,
            "test_type": test_type,
            "coverage": coverage
        })

    async def validate_code_quality(self, code_path: str, quality_gates: List[str]) -> MCPToolResult:
        """Validate code quality and standards (QA Agent only)"""
        if not self.client.is_tool_available("validate_code_quality"):
            return MCPToolResult(success=False, error="Tool not available for this agent", tool_name="validate_code_quality")

        return await self.client.execute_tool("validate_code_quality", {
            "code_path": code_path,
            "quality_gates": quality_gates
        })

    async def security_scan(self, target_path: str, scan_type: str) -> MCPToolResult:
        """Perform security vulnerability scan (QA Agent only)"""
        if not self.client.is_tool_available("security_scan"):
            return MCPToolResult(success=False, error="Tool not available for this agent", tool_name="security_scan")

        return await self.client.execute_tool("security_scan", {
            "target_path": target_path,
            "scan_type": scan_type
        })

    # ========================================================================
    # Scheduler Agent Specific Tools
    # ========================================================================

    async def optimize_schedule(self, tasks: List[Dict], constraints: Dict, optimization_goal: str) -> MCPToolResult:
        """Optimize task schedule (Scheduler Agent only)"""
        if not self.client.is_tool_available("optimize_schedule"):
            return MCPToolResult(success=False, error="Tool not available for this agent", tool_name="optimize_schedule")

        return await self.client.execute_tool("optimize_schedule", {
            "tasks": tasks,
            "constraints": constraints,
            "optimization_goal": optimization_goal
        })

    async def check_resource_availability(self, resource_type: str, time_window: Dict) -> MCPToolResult:
        """Check resource availability (Scheduler Agent only)"""
        if not self.client.is_tool_available("check_resource_availability"):
            return MCPToolResult(success=False, error="Tool not available for this agent", tool_name="check_resource_availability")

        return await self.client.execute_tool("check_resource_availability", {
            "resource_type": resource_type,
            "time_window": time_window
        })

    async def forecast_capacity(self, historical_data: List[float], forecast_period: int) -> MCPToolResult:
        """Forecast future capacity needs (Scheduler Agent only)"""
        if not self.client.is_tool_available("forecast_capacity"):
            return MCPToolResult(success=False, error="Tool not available for this agent", tool_name="forecast_capacity")

        return await self.client.execute_tool("forecast_capacity", {
            "historical_data": historical_data,
            "forecast_period": forecast_period
        })

    # ========================================================================
    # Project Manager Agent Specific Tools
    # ========================================================================

    async def generate_report(self, report_type: str, data_sources: List[str], format: str) -> MCPToolResult:
        """Generate comprehensive project report (Project Manager Agent only)"""
        if not self.client.is_tool_available("generate_report"):
            return MCPToolResult(success=False, error="Tool not available for this agent", tool_name="generate_report")

        return await self.client.execute_tool("generate_report", {
            "report_type": report_type,
            "data_sources": data_sources,
            "format": format
        })

    async def update_stakeholders(self, stakeholders: List[str], update_type: str, content: Dict) -> MCPToolResult:
        """Update project stakeholders (Project Manager Agent only)"""
        if not self.client.is_tool_available("update_stakeholders"):
            return MCPToolResult(success=False, error="Tool not available for this agent", tool_name="update_stakeholders")

        return await self.client.execute_tool("update_stakeholders", {
            "stakeholders": stakeholders,
            "update_type": update_type,
            "content": content
        })

    async def risk_assessment(self, project_data: Dict, risk_categories: List[str]) -> MCPToolResult:
        """Perform comprehensive risk assessment (Project Manager Agent only)"""
        if not self.client.is_tool_available("risk_assessment"):
            return MCPToolResult(success=False, error="Tool not available for this agent", tool_name="risk_assessment")

        return await self.client.execute_tool("risk_assessment", {
            "project_data": project_data,
            "risk_categories": risk_categories
        })

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def get_available_tools_for_agent(self) -> List[str]:
        """Get list of tools available for this agent"""
        available = []
        for tool_name, tool_info in self.client.available_tools.items():
            agent_access = tool_info.get("agent_access", [])
            if self.agent_name in agent_access or "all" in agent_access:
                available.append(tool_name)
        return available

    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific tool"""
        return self.client.available_tools.get(tool_name)

    async def list_available_tools(self) -> Dict[str, Any]:
        """List all tools available to this agent with descriptions"""
        available_tools = {}
        for tool_name in self.get_available_tools_for_agent():
            tool_info = self.get_tool_info(tool_name)
            if tool_info:
                available_tools[tool_name] = {
                    "description": tool_info.get("description", ""),
                    "type": tool_info.get("type", ""),
                    "parameters": tool_info.get("parameters", {}),
                    "returns": tool_info.get("returns", {})
                }
        return available_tools

# ============================================================================
# Demo and Testing Functions
# ============================================================================

async def demo_mcp_client():
    """Demonstrate MCP client functionality"""
    print("🤖 MCP Client Demo")
    print("=" * 50)

    # Test with different agent types
    agents = ["qa", "scheduler", "project_manager"]

    for agent_name in agents:
        print(f"\n🔧 Testing agent: {agent_name}")
        print("-" * 30)

        integration = MCPToolsIntegration(agent_name=agent_name)

        # Initialize
        if await integration.initialize():
            print(f"✅ Connected as {agent_name}")

            # List available tools
            tools = await integration.list_available_tools()
            print(f"📋 Available tools: {list(tools.keys())}")

            # Test general tools
            time_result = await integration.get_current_time()
            if time_result.success:
                print(f"⏰ Current time: {time_result.result['human_readable']}")

            # Test metrics calculation
            metrics_result = await integration.calculate_metrics([1, 2, 3, 4, 5], ["mean", "std"])
            if metrics_result.success:
                print(f"📊 Metrics: {metrics_result.result['metrics']}")

            # Test agent-specific tools
            if agent_name == "qa":
                test_result = await integration.run_test_suite("/tests", "unit", True)
                if test_result.success:
                    print(f"🧪 Test results: {test_result.result['execution_summary']}")

            elif agent_name == "scheduler":
                schedule_result = await integration.optimize_schedule(
                    [{"name": "Task 1", "duration": 8}, {"name": "Task 2", "duration": 4}],
                    {"budget": 10000},
                    "minimize_time"
                )
                if schedule_result.success:
                    print(f"📅 Schedule optimized: {schedule_result.result['efficiency_gain']} efficiency gain")

            elif agent_name == "project_manager":
                report_result = await integration.generate_report("status", ["database", "api"], "json")
                if report_result.success:
                    print(f"📄 Report generated: {report_result.result['metadata']}")

            # Shutdown
            await integration.shutdown()

        else:
            print(f"❌ Failed to connect as {agent_name}")

        await asyncio.sleep(1)

async def interactive_client_demo():
    """Interactive MCP client demo"""
    print("🤖 Interactive MCP Client Demo")
    print("=" * 50)

    agent_name = input("Enter agent name (qa/scheduler/project_manager): ").strip()
    if not agent_name:
        agent_name = "qa"

    integration = MCPToolsIntegration(agent_name=agent_name)

    if await integration.initialize():
        print(f"✅ Connected as {agent_name}")
        print("Available commands:")
        print("  time - Get current time")
        print("  metrics - Calculate sample metrics")
        print("  tools - List available tools")
        print("  exit - Quit demo")

        while True:
            try:
                command = input(f"\n[{agent_name}] Enter command: ").strip().lower()

                if command == "exit":
                    break
                elif command == "time":
                    result = await integration.get_current_time()
                    if result.success:
                        print(f"⏰ {result.result}")
                    else:
                        print(f"❌ {result.error}")

                elif command == "metrics":
                    result = await integration.calculate_metrics([1, 2, 3, 4, 5], ["mean", "median", "stdev"])
                    if result.success:
                        print(f"📊 {result.result}")
                    else:
                        print(f"❌ {result.error}")

                elif command == "tools":
                    tools = await integration.list_available_tools()
                    print("🔧 Available tools:")
                    for tool_name, tool_info in tools.items():
                        print(f"  • {tool_name}: {tool_info['description']}")

                else:
                    print("❓ Unknown command")

            except KeyboardInterrupt:
                break

        await integration.shutdown()
        print("👋 Goodbye!")
    else:
        print("❌ Failed to connect to MCP server")

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="MCP Client Demo")
    parser.add_argument("--mode", choices=["demo", "interactive"], default="demo", help="Demo mode")
    parser.add_argument("--agent", default="qa", help="Agent name")
    parser.add_argument("--server", default="http://localhost:8080", help="MCP server URL")

    args = parser.parse_args()

    try:
        if args.mode == "demo":
            asyncio.run(demo_mcp_client())
        elif args.mode == "interactive":
            asyncio.run(interactive_client_demo())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()