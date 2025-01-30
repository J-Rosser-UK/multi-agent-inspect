import asyncio

from base import Agent, Meeting, Chat, Wrapper

from sqlalchemy.orm import Session


class StepBackAgentSystem:
    def __init__(self, session: Session):
        self.Agent = Wrapper(Agent, session)
        self.Meeting = Wrapper(Meeting, session)
        self.Chat = Wrapper(Chat, session)
        self.session = session

    async def forward(self, task: str) -> str:
        # Create agents
        system = self.Agent(agent_name="system", temperature=0.8)
        principle_agent = self.Agent(agent_name="Principle Agent", temperature=0.8)
        cot_agent = self.Agent(agent_name="Chain-of-Thought Agent", temperature=0.8)

        # Setup meeting
        meeting = self.Meeting(meeting_name="step_back_meeting")
        meeting.agents.extend([system, principle_agent, cot_agent])

        # First get the principles involved
        meeting.chats.append(
            self.Chat(
                agent=system,
                content="What are the physics, chemistry or biology principles and concepts involved in solving this task? First think step by step. Then list all involved principles and explain them.",
            )
        )

        principle_output = await principle_agent.forward(
            response_format={
                "thinking": "Your step by step thinking about the principles.",
                "principles": "List and explanation of the principles involved.",
            }
        )

        meeting.chats.append(
            self.Chat(
                agent=principle_agent,
                content=principle_output["thinking"] + principle_output["principles"],
            )
        )

        # Now solve using the principles
        meeting.chats.append(
            self.Chat(
                agent=system,
                content=f"Given the question and the involved principles above, think step by step and then solve the task: {task}",
            )
        )

        final_output = await cot_agent.forward(
            response_format={
                "thinking": "Your step by step thinking.",
                "answer": "A single letter, A, B, C or D.",
            }
        )

        return final_output["answer"]


if __name__ == "__main__":
    from base import initialize_session

    session, Base = initialize_session
    agent_system = StepBackAgentSystem()
    task = "What should I have for dinner?A: soup B: burgers C: pizza D: pasta"
    output = asyncio.run(agent_system.forward(task))
    print(output)
