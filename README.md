<div align="center">

  <h1>AISI Multi-Agent Inspect</h1>
  
  <p>
    Expanding AISI's Inspect for robust multi-agent framework evals.
  </p>
  
  
<!-- Badges -->
<p>
  <a href="https://github.com/J-Rosser-UK/multi-agent-inspect/contributors">
    <img src="https://img.shields.io/github/contributors/J-Rosser-UK/multi-agent-inspect" alt="contributors" />
  </a>
  <a href="">
    <img src="https://img.shields.io/github/last-commit/J-Rosser-UK/multi-agent-inspect" alt="last update" />
  </a>
  <a href="https://github.com/J-Rosser-UK/multi-agent-inspect/network/members">
    <img src="https://img.shields.io/github/forks/J-Rosser-UK/multi-agent-inspect" alt="forks" />
  </a>
  <a href="https://github.com/J-Rosser-UK/multi-agent-inspect/stargazers">
    <img src="https://img.shields.io/github/stars/J-Rosser-UK/multi-agent-inspect" alt="stars" />
  </a>
  <a href="https://github.com/J-Rosser-UK/multi-agent-inspect/issues/">
    <img src="https://img.shields.io/github/issues/J-Rosser-UK/multi-agent-inspect" alt="open issues" />
  </a>
  <a href="https://github.com/J-Rosser-UK/multi-agent-inspect/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/J-Rosser-UK/multi-agent-inspect.svg" alt="license" />
  </a>
</p>
   
<h4>
    </span>
    <a href="https://github.com/J-Rosser-UK/multi-agent-inspect/issues/">Report Bug</a>
  <span> · </span>
    <a href="https://github.com/J-Rosser-UK/multi-agent-inspect/issues/">Request Feature</a>
  </h4>
</div>

<br />


High-performing agent scaffolding is crucial for assessing the upper limits of LLM capabilities, such that during evals we do not underestimate their potential. Multi-agent systems are known to generally outperform single agents on eval benchmarks, therefore this approach expands inspect to be capable of evaluating a wide range of multi-agent systems defined via a simple communication protocol.

This scaffolding was developed in conjunction with a larger piece of research of mine, where a Meta Agent is capable of designing novel Multi-Agent systems to solve tasks by expressing them as runnable python code. This research expands upon the [ADAS paper](https://arxiv.org/abs/2408.08435).

## Multi-Agent Frameworks

In this repo, I have implemented the 7 example multi-agent frameworks given in the [ADAS paper](https://arxiv.org/abs/2408.08435) in my new communication protocol in `/examples`. 

Communication is acheived via the observer design pattern. Agents can subscribe to meetings, and then "Chat" to publish some text to that meeting.

1. `Agent` - This is a simple single agent, who can only respond in JSON format.
2. `Meeting` - Only agents participating ("subscribing") in the same meeting can hear eachother. An agent can be in multiple meetings and their chat history is stored chronologically.
3. `Chat` - Agents can only "speak" ("publish") in meetings if they "Chat", otherwise if they carry out a task it will not be visible to the other agents *or themselves*. It may sometimes be necessary to instantiate an internal dialogue meeting so that the agent can think by itself.

```python
async def forward(self, task: str) -> str:
    # Create a system agent to provide instructions
    system = self.Agent(agent_name="system", temperature=0.8)

    # Create the Chain-of-Thought agent
    cot_agent = self.Agent(agent_name="Chain-of-Thought Agent", temperature=0.7)

    # Setup meeting
    meeting = self.Meeting(meeting_name="chain-of-thought")
    meeting.agents.extend([system, cot_agent])

    # Add system instruction
    meeting.chats.append(
        self.Chat(
            agent=system,
            content=f"Please think step by step and then solve the task: {task}",
        )
    )

    # Get response from COT agent
    output = await cot_agent.forward(
        response_format={
            "thinking": "Your step by step thinking.",
            "answer": "A single letter, A, B, C or D.",
        }
    )

    # Record the agent's response in the meeting
    meeting.chats.append(self.Chat(agent=cot_agent, content=output["thinking"]))

    return output["answer"]
```

## Extras

### Avoiding rate limits via throttlling

When evaluating multi-agent systems in parallel, you can easily experience rate limit errors. In particular if you flood a million API requests in parallel, they'll not just exceed the rate limits but can also fail with errors. Therefore I implemented throttling through a fastapi in `chat/api.py`.

### SQLAlchemy for Object Storage

All objects are stored in a sqlite database. This code was first implemented in my multi-agent eval research however I found it useful to maintain it here.


## TODO

### Better batching
At the moment only one multi-agent framework can be evaluated at a time (although the eval is parallelised). It should be fairly straightforward to implement the evaluation of a batch of frameworks at once.

### Implementing an Open-Source multi-agent framework
At the moment I'm using a custom multi-agent framework, to ensure that everything is non-blocking. However it would be work exploring something like LangGraph in the future.

## Installation


### Setup

We use [uv](https://docs.astral.sh/uv/getting-started/installation/) for package
management (as an alternative to tools such as `pip` or `poetry`). `uv` is a bit different
to `pip` in that you don't need to setup a virtual environment yourself or manage
specific Python versions - it does all that for you. We also use
[PoeThePoet](https://poethepoet.natn.io/installation.html) for common commands (running
tests, linting, etc.). You can follow their installation instructions, or alternatively
if you already have `pipx` installed, you can run:

```bash
pipx install uv poethepoet
```

Once you've done that, you can install all packages with the following command (this
will create a virtual environment for you and install the dependencies):

```bash
uv sync --dev
```

Once installed, use `uv run python ./path/to/script.py` to run python files.
