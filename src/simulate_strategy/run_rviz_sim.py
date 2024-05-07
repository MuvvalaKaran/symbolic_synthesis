import time
import yaml
import warnings

from subprocess import Popen, PIPE

from config import SIM_CONFIG_PATH

class run_rviz_sim():

    def __init__(self):
        with open(SIM_CONFIG_PATH, 'r') as file:
            self.sim_obj_config = yaml.safe_load(file)
    
    # @property
    # def sim_handle(self):
    #     return self._sim_handle
    

    def send_command(self, act_name: str, loc: str = ''):
        """
         A wrapper function that call the corresponding action primitive based on the action based.
         Note: act_name can be robot action or human action.
        """
        if 'transit' in act_name:
            # cmd_handle = Popen(['rosservice', 'call', '/manipulator_node/action_primitive/linear_transit', final_string], stdout=PIPE)
            self.send_transit_command_to_robot(act_name=act_name, loc=loc)
        elif 'transfer' in act_name:
            # cmd_handle = Popen(['rosservice call /manipulator_node/action_primitive/linear_transport ' + f'{loc}'], shell=True, stdout=PIPE)
            self.send_transport_command_to_robot(act_name=act_name)
        elif 'release' in act_name:
            # cmd_handle = Popen(["rosservice call /manipulator_node/action_primitive/release '' "], shell=True, stdout=PIPE)
            self.send_release_command_to_robot()
        elif 'grasp' in act_name:
            self.send_grasp_command_to_robot()
            # cmd_handle = Popen(["rosservice call /manipulator_node/action_primitive/grasp '' "], shell=True, stdout=PIPE)
        elif 'human' in act_name:
            # cmd_handle = Popen(['rosservice call /manipulator_node/action_primitive/set_object_locations' + f' [{box}] ' + f'[{loc}]'], shell=True, stdout=PIPE)
            self.send_human_move_command(act_name=act_name)


    # (raw_output, err) = cmd_handle.communicate()
    # if err:
    #     print(err)


    # def stow_robot() -> bool:
    # 	# helper function that updated the status (X, Y, Z) of the all the objects from vicon
    # 	rospy.wait_for_service("/manipulator_node/action_primitive/stow")

    # 	rospy.ServiceProxy("/manipulator_node/action_primitive/stow", Stow)()
        


    # def update_env_status() -> bool:
    # 	# helper function that motion plans and stows the robot
    # 	rospy.wait_for_service("/manipulator_node/action_primitive/update_environment")

    # 	update_handle = rospy.ServiceProxy("/manipulator_node/action_primitive/update_environment", UpdateEnv)
    # 	t = update_handle(False)


    def send_transport_command_to_robot(self, act_name: str) -> bool:
        # #  convenience method that blocks until the service named is available
        # rospy.wait_for_service("/manipulator_node/action_primitive/linear_transport")

        # # create a handle for calling the service
        # transport_handle = rospy.ServiceProxy("/manipulator_node/action_primitive/linear_transport", Transit)
        # t = transport_handle(loc)
        # return t.plan_success
        loc = act_name.split()[1]
        cmd_handle = Popen(['rosservice call /manipulator_node/action_primitive/linear_transport ' + f'{loc}'], shell=True, stdout=PIPE)

        (raw_output, err) = cmd_handle.communicate()
        if err:
            print(err)



    def send_transit_command_to_robot(self, act_name: str, loc: str) -> bool:
        #  convenience method that blocks until the service named is available
        final_string: str = f"destination_location: '{loc}'"
        cmd_handle = Popen(['rosservice', 'call', '/manipulator_node/action_primitive/linear_transit', final_string], stdout=PIPE)

        (raw_output, err) = cmd_handle.communicate()
        if err:
            print(err)
        
        

    def send_grasp_command_to_robot(self) -> bool:
        # #  convenience method that blocks until the service named is available
        # rospy.wait_for_service("/manipulator_node/action_primitive/grasp")

        # # create a handle for calling the service
        # grasp_handle = rospy.ServiceProxy("/manipulator_node/action_primitive/grasp", Grasp)
        # t = grasp_handle(obj_id)
        # return t.mv_props.execution_success
        cmd_handle = Popen(["rosservice call /manipulator_node/action_primitive/grasp '' "], shell=True, stdout=PIPE)

        (raw_output, err) = cmd_handle.communicate()
        if err:
            print(err)


    def send_release_command_to_robot(self) -> bool:
        #  convenience method that blocks until the service named is available
        # rospy.wait_for_service("/manipulator_node/action_primitive/grasp")

        # # create a handle for calling the service
        # release_handle = rospy.ServiceProxy("/manipulator_node/action_primitive/release", Release)
        # t = release_handle(obj_id)
        # return t.mv_props.execution_success
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