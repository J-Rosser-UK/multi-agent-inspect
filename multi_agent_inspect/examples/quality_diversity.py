import asyncio

from base import Agent, Meeting, Chat, Wrapper

from sqlalchemy.orm import Session


class QDAgentSystem:
    def __init__(self, session: Session):
        self.Agent = Wrapper(Agent, session)
        self.Meeting = Wrapper(Meeting, session)
        self.Chat = Wrapper(Chat, session)
        self.session = session

    async def forward(self, task: str) -> str:
        # Create agents
        system = self.Agent(agent_name="system", temperature=0.8)
        cot_agent = self.Agent(agent_name="Chain-of-Thought Agent", temperature=0.8)
        final_decision_agent = self.Agent(
            agent_name="Final Decision Agent", temperature=0.1
        )

        # Setup meeting
        meeting = self.Meeting(meeting_name="quality_diversity_meeting")
        meeting.agents.extend([system, cot_agent, final_decision_agent])

        N_max = 3  # Maximum number of attempts

        # Initial attempt
        meeting.chats.append(
            self.Chat(
                agent=system,
                content=f"Please think step by step and then solve the task: {task}",
            )
        )

        output = await cot_agent.forward(
            response_format={
                "thinking": "Your step by step thinking.",
                "answer": "A single letter, A, B, C or D.",
            }
        )

        meeting.chats.append(
            self.Chat(agent=cot_agent, content=output["thinking"] + output["answer"])
        )

        # Generate diverse solutions
        for i in range(N_max):
            meeting.chats.append(
                self.Chat(
                    agent=system,
                    content=f"Given previous attempts, try to come up with another interesting way to solve the task: {task}",
                )
            )

            output = await cot_agent.forward(
                response_format={
                    "thinking": "Your step by step thinking with a new approach.",
                    "answer": "A single letter, A, B, C or D.",
                }
            )

            meeting.chats.append(
                self.Chat(
                    agent=cot_agent, content=output["thinking"] + output["answer"]
                )
            )

        # Make final decision
        meeting.chats.append(
            self.Chat(
                agent=system,
                content="Given all the above solutions, reason over them carefully and provide a final answer.",
            )
        )

        final_output = await final_decision_agent.forward(
            response_format={
                "thinking": "Your step by step thinking comparing all solutions.",
                "answer": "A single letter, A, B, C or D.",
            }
        )

        return final_output["answer"]


if __name__ == "__main__":
    from base import initialize_session

    session, Base = initialize_session
    agent_system = QDAgentSystem()
    task = "What should I have for dinner?A: soup B: burgers C: pizza D: pasta"
    output = asyncio.run(agent_system.forward(task))
    print(output)
