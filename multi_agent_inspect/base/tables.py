from sqlalchemy import Column, String, DateTime, ForeignKey, Float, JSON
from sqlalchemy.orm import relationship
import datetime
import uuid
import random
import string
from sqlalchemy.orm import object_session
from .base import CustomBase, CustomColumn, AutoSaveList
from chat import get_structured_json_response_from_gpt
import asyncio
from functools import wraps
import threading


class Chat(CustomBase):
    __tablename__ = "chat"

    chat_id = CustomColumn(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        label="The chat's unique identifier (UUID).",
    )
    agent_id = CustomColumn(
        String, ForeignKey("agent.agent_id"), label="The role of the chat."
    )
    meeting_id = CustomColumn(
        String,
        ForeignKey("meeting.meeting_id"),
        label="The meeting's unique identifier (UUID).",
    )
    content = CustomColumn(String, label="The content of the chat.")
    chat_timestamp = CustomColumn(
        DateTime, default=datetime.datetime.utcnow, label="The timestamp of the chat."
    )

    # Relationships
    agent = relationship("Agent", back_populates="chats", collection_class=AutoSaveList)
    meeting = relationship(
        "Meeting", back_populates="chats", collection_class=AutoSaveList
    )


class Meeting(CustomBase):
    __tablename__ = "meeting"

    meeting_id = CustomColumn(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        label="The chat's unique identifier (UUID).",
    )
    meeting_name = CustomColumn(String, label="The name of the meeting.")
    meeting_timestamp = CustomColumn(
        DateTime,
        default=datetime.datetime.utcnow,
        label="The timestamp of the meeting.",
    )

    # Relationships

    chats = relationship(
        "Chat", back_populates="meeting", collection_class=AutoSaveList
    )
    agents = relationship(
        "Agent",
        secondary="agents_by_meeting",
        back_populates="meetings",
        collection_class=AutoSaveList,
    )


class AgentsbyMeeting(CustomBase):
    __tablename__ = "agents_by_meeting"

    agent_id = CustomColumn(
        String,
        ForeignKey("agent.agent_id"),
        primary_key=True,
        label="The agent's unique identifier (UUID).",
    )
    meeting_id = CustomColumn(
        String,
        ForeignKey("meeting.meeting_id"),
        primary_key=True,
        label="The chat's unique identifier (UUID).",
    )
    agents_by_meeting_timestamp = CustomColumn(
        DateTime,
        default=datetime.datetime.utcnow,
        label="The timestamp of the agent's addition to the meeting.",
    )


class Agent(CustomBase):
    __tablename__ = "agent"

    agent_id = CustomColumn(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        label="The agent's unique identifier (UUID).",
    )
    agent_name = CustomColumn(String, label="The agent's name.")
    agent_backstory = CustomColumn(
        String, label="A long description of the agent's backstory."
    )
    model = CustomColumn(String, label="The LLM model to be used.")
    temperature = CustomColumn(
        Float,
        default=0.7,
        label="The sampling temperature. The higher the temperature, the more creative the responses.",
    )
    agent_timestamp = CustomColumn(
        DateTime,
        default=datetime.datetime.utcnow,
        label="The timestamp of the agent's creation.",
    )

    # Relationships
    chats = relationship("Chat", back_populates="agent", collection_class=AutoSaveList)
    meetings = relationship(
        "Meeting",
        secondary="agents_by_meeting",
        back_populates="agents",
        collection_class=AutoSaveList,
    )

    def __init__(self, session, agent_name, model="gpt-4o-mini", temperature=0.5):
        super().__init__(
            session, agent_name=agent_name, model=model, temperature=temperature
        )
        characters = (
            string.ascii_letters + string.digits
        )  # includes both upper/lower case letters and numbers
        random_id = "".join(random.choices(characters, k=4))
        self.agent_name = agent_name + " " + random_id

    def __repr__(self):
        return f"{self.agent_name} {self.agent_id}"

    @property
    def chat_history(self):
        meetings = self.meetings
        chats = []
        for meeting in meetings:
            chats.extend(meeting.chats)

        # order chats by timestamp
        chats = sorted(chats, key=lambda x: x.chat_timestamp)

        # Convert into the format [{role: agent, content: chat_content}]
        def to_chat(chat):
            chat_content: str = chat.content if chat.content else ""

            if chat.agent.agent_id == self.agent_id:
                role = "assistant"
                content = "You: " + chat_content
            elif chat.agent.agent_name == "system":
                role = "system"
                content = "System: " + chat_content

            else:
                role = "user"
                content = chat.agent.agent_name + ": " + chat_content

            return {"role": role, "content": content}

        chats = [to_chat(chat) for chat in chats]
        return chats

    async def forward(self, response_format) -> dict:

        # logging.info(f"Agent {self.agent_name} is thinking...")

        messages = self.chat_history

        response_json = await get_structured_json_response_from_gpt(
            messages=messages, response_format=response_format, temperature=0.5
        )

        # logging.info(f"Agent {self.agent_name} has responded with: \n{response_json}\n -------------------")

        return response_json
