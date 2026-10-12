"""Logical agent registry and request router."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .context_selection import AgentContextManifest


class AgentRole(str, Enum):
    PLANNER = "planner_agent"
    EXECUTOR = "executor_agent"
    REPLANNER = "replanner_agent"
    SPECIALIST = "specialist_agent"


@dataclass
class LogicalAgent:
    agent_id: str
    task_id: str
    role: AgentRole
    model_endpoint: str
    token_budget: int
    tool_permissions: tuple[str, ...] = ()
    model_policy: str = "backend"
    mutable_history: list[dict[str, Any]] = field(default_factory=list)
    selected_context_manifest: AgentContextManifest | None = None


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, LogicalAgent] = {}

    def register(self, agent: LogicalAgent) -> None:
        if agent.agent_id in self._agents:
            raise ValueError(f"duplicate agent_id {agent.agent_id}")
        self._agents[agent.agent_id] = agent

    def get(self, agent_id: str) -> LogicalAgent:
        return self._agents[agent_id]

    def find(self, task_id: str, role: AgentRole) -> list[LogicalAgent]:
        return [agent for agent in self._agents.values() if agent.task_id == task_id and agent.role == role]


class AgentRouter:
    _STAGE_ROLE = {
        "planner": AgentRole.PLANNER,
        "plan": AgentRole.PLANNER,
        "execute": AgentRole.EXECUTOR,
        "tool": AgentRole.EXECUTOR,
        "replanner": AgentRole.REPLANNER,
        "replan": AgentRole.REPLANNER,
    }

    def __init__(self, registry: AgentRegistry) -> None:
        self.registry = registry

    def route(
        self,
        *,
        task_id: str,
        loop_stage: str,
        task_type: str = "",
        objective: str = "",
        required_tools: tuple[str, ...] = (),
        model_policy: str | None = None,
    ) -> LogicalAgent:
        role = self._STAGE_ROLE.get(loop_stage.lower(), AgentRole.SPECIALIST)
        candidates = self.registry.find(task_id, role)
        if model_policy:
            candidates = [agent for agent in candidates if agent.model_policy == model_policy]
        if required_tools:
            required = set(required_tools)
            candidates = [agent for agent in candidates if required <= set(agent.tool_permissions)]
        if not candidates:
            raise LookupError(f"no logical agent for task={task_id} stage={loop_stage}")
        return sorted(candidates, key=lambda agent: agent.agent_id)[0]
