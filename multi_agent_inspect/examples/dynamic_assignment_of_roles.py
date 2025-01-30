import asyncio

from base import Agent, Meeting, Chat, Wrapper

from sqlalchemy.orm import Session


class DynamicRolesAgentSystem:
    def __init__(self, session: Session):
        self.Agent = Wrapper(Agent, session)
        self.Meeting = Wrapper(Meeting, session)
        self.Chat = Wrapper(Chat, session)
        self.session = session

    async def forward(self, task: str) -> str:
        # Create agents
        system = self.Agent(agent_name="system", temperature=0.8)
        routing_agent = self.Agent(agent_name="Routing Agent", temperature=0.8)

        expert_agents = {
            "physics": self.Agent(agent_name="Physics Expert", temperature=0.8),
            "chemistry": self.Agent(agent_name="Chemistry Expert", temperature=0.8),
            "biology": self.Agent(agent_name="Biology Expert", temperature=0.8),
            "general": self.Agent(agent_name="Science Generalist", temperature=0.8),
        }

        # Setup meeting
        meeting = self.Meeting(meeting_name="role_assignment_meeting")
        meeting.agents.extend([system, routing_agent] + list(expert_agents.values()))

        # Route the task
        meeting.chats.append(
            self.Chat(
                agent=system,
                content="Given the task, please choose an Expert to answer the question. Choose from: Physics, Chemistry, Biology Expert, or Science Generalist.",
            )
        )

        routing_output = await routing_agent.forward(
            response_format={
                "choice": "One of: physics, chemistry, biology, or general"
            }
        )

        # Select expert based on routing decision
        expert_choice = routing_output["choice"].lower()
        if expert_choice not in expert_agents:
            expert_choice = "general"

        selected_expert = expert_agents[expert_choice]

        # Get answer from selected expert
        meeting.chats.append(
            self.Chat(
                agent=system,
                content=f"Please think step by step and then solve the task: {task}",
            )
        )

        expert_output = await selected_expert.forward(
            response_format={
                "thinking": "Your step by step thinking.",
                "answer": "A single letter, A, B, C or D.",
            }
        )

        return expert_output["answer"]


if __name__ == "__main__":
    from base import initialize_session

    session, Base = initialize_session
    agent_system = DynamicRolesAgentSystem()
    task = "What should I have for dinner?A: soup B: burgers C: pizza D: pasta"
    output = asyncio.run(agent_system.forward(task))
    print(output)
