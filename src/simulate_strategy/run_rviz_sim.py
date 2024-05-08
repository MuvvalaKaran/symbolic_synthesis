import re
import time
import yaml
import warnings

from subprocess import Popen, PIPE

from config import SIM_CONFIG_PATH

class run_rviz_sim():

    def __init__(self):
        with open(SIM_CONFIG_PATH, 'r') as file:
            self.sim_obj_config = yaml.safe_load(file)
    

    def send_command(self, act_name: str, loc: str = ''):
        """
         A wrapper function that call the corresponding action primitive based on the action based.
         Note: act_name can be robot action or human action.
        """
        if 'transit' in act_name:
            box_number: int = re.search(r"b(\d+)", act_name).group(1)
            self.send_transit_command_to_robot(box=f'b{box_number}', loc=loc)
        elif 'transfer' in act_name:
            self.send_transport_command_to_robot(act_name=act_name)
        elif 'release' in act_name:
            self.send_release_command_to_robot()
        elif 'grasp' in act_name:
            self.send_grasp_command_to_robot()
        elif 'human' in act_name:
            self.send_human_move_command(act_name=act_name)


    def send_transport_command_to_robot(self, act_name: str) -> bool:
        loc = act_name.split()[1]
        cmd_handle = Popen(['rosservice call /manipulator_node/action_primitive/linear_transport ' + f'{loc}'], shell=True, stdout=PIPE)

        (raw_output, err) = cmd_handle.communicate()
        if err:
            print(err)

    def send_transit_command_to_robot(self, box: str, loc: str) -> bool:
        final_string: str = f"destination_location: '{loc}'"
        if loc == 'else':
            true_loc = self.sim_obj_config[box]['initial_location']
            final_string = f"destination_location: '{true_loc}'" 
        
        cmd_handle = Popen(['rosservice', 'call', '/manipulator_node/action_primitive/linear_transit', final_string], stdout=PIPE)

        (raw_output, err) = cmd_handle.communicate()
        if err:
            print(err)
        
        

    def send_grasp_command_to_robot(self) -> bool:
        cmd_handle = Popen(["rosservice call /manipulator_node/action_primitive/grasp '' "], shell=True, stdout=PIPE)

        (raw_output, err) = cmd_handle.communicate()
        if err:
            print(err)


    def send_release_command_to_robot(self) -> bool:
        cmd_handle = Popen(["rosservice call /manipulator_node/action_primitive/release '' "], shell=True, stdout=PIPE)

        (raw_output, err) = cmd_handle.communicate()
        if err:
            print(err)

        return True


    def send_human_move_command(self, act_name: str) -> bool:
        """
        Send set object location. 
        """
        box, loc = act_name.split()[1], act_name.split()[2]
        cmd_handle = Popen(['rosservice call /manipulator_node/action_primitive/set_object_locations' + f' [{box}] ' + f'[{loc}]'], shell=True, stdout=PIPE)
        (raw_output, err) = cmd_handle.communicate()
        if err:
            print(err)

    def spin_up_sim(self):
        """
         Helper method to spin up RVIZ sim
        """
        self._sim_handle = Popen(["roslaunch", "taskit", "manipulator_node.launch"])
        # wait for the sim to launch
        time.sleep(30)
    
    def terminate_sim(self):
        """
         Warpaper around shutdown sim command.
        """
        self._sim_handle.terminate()


# if  __name__ == "__main__":
#     spin_up_sim()