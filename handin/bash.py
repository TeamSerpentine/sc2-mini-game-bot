
import os
import subprocess


class RunBash:
    def __init__(self, path_to_bot=os.getcwd(), module=None, bot_name=None):
        self.path_to_bot = path_to_bot
        self.module = module
        self.bot_name = bot_name
        self.bash_command = "python -m pysc2.bin.agent --map CompetitionMap --agent {}.{}"

    def run_bash(self):
        """ Will run the simple agent script from FruitpunchAI.  """
        assert self.module is not None, "No module name provided for the bot class."
        assert self.bot_name is not None, "No class name specified for the bot."

        bashCommand = self.bash_command.format(self.module, self.bot_name)
        process = subprocess.Popen(bashCommand.split(), cwd=self.path_to_bot, stdout=subprocess.PIPE)
        output, error = process.communicate()
        print(f"\nOuput: {output}\n\nError: {error}")

    def run_bash_with_features(self):
        """ adds the features units before running the bash command.  """
        self.bash_command += " --use_feature_units"
        self.run_bash()

    def run_bash_command(self, command, cwd=os.getcwd(), show_output=False):
        """ Will run any batch command that you enter in command.

            command: string
                The code that has to be run inside the command prompt
                you can add multiple commands by using the '&&' between them
            cwd: string
                The location where the command prompt will start from
        """
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, cwd=cwd)
        if not show_output:
            process.communicate()
            return

        output, error = process.communicate()
        print(f"\nOuput: {output}\n\nError: {error}")


if __name__ == "__main__":
    # Example for running the bash_script
    agent_extreme = "extreme.ExtremeBackUp"

    agent_1 = "danger_noodle.Serpentine"
    agent_1_2 = "danger_noodle.SerpentineV2"

    agent_2 = "danger_noodle_back_up.Serpentine"
    agent_2_2 = "danger_noodle_back_up.SerpentineV2"

    bash = RunBash()
    bash.run_bash_command(f"python -m pysc2.bin.agent --map CompetitionMap "
                          f"--max_episodes 1  --use_feature_units "
                          f"--agent {agent_1} --agent2 {agent_1_2} ")
