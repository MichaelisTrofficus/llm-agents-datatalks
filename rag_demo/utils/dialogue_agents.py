from typing import List

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage
)


class DialogueAgent:

    def __init__(
        self,
        name,
        system_message: SystemMessage,
        model: ChatOpenAI,
    ) -> None:
        self.name = name
        self.system_message = system_message
        self.model = model
        self.message_history = f"""Here is the conversation so far.
        """
        self.prefix = f'\n{self.name}:'

    def send(self) -> str:
        """
        Applies the chatmodel to the message history
        and returns the message string
        """
        message = self.model(
            [self.system_message,
             HumanMessage(content=self.message_history+self.prefix)])
        return message.content

    def receive(self, name: str, message: str) -> None:
        """
        Concatenates {message} spoken by {name} into message history
        """
        self.message_history += f'\n{name}: {message}'


class DialogueSimulator():

    def __init__(self, agents: List[DialogueAgent]):
        self.agents = agents
        self._step = 0

    def reset(self, name: str, message: str):
        """
        Initiates the conversation with a {message} from {name}
        """
        for agent in self.agents:
            agent.receive(name, message)

    def select_next_speaker(self, step: int) -> int:
        idx = (step + 1) % len(self.agents)
        return idx

    def step(self) -> tuple[str, str]:
        speaker = self.agents[self.select_next_speaker(self._step)]
        message = speaker.send()
        for receiver in self.agents:
            receiver.receive(speaker.name, message)
        self._step += 1
        return speaker.name, message


def run_simulation(num_rounds, moderator, objective, dialogue_agents):
    max_iters = num_rounds
    n = 0

    simulator = DialogueSimulator(agents=dialogue_agents)
    simulator.reset(moderator, objective)
    print(f"({moderator}): {objective}")
    print('\n')

    while n < max_iters:
        name, message = simulator.step()
        print(f"({name}): {message}")
        print('\n')
        n += 1
