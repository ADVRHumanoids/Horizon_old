from Horizon.horizon import *

class footsteps_scheduler(constraint_handler):
    """
    The foostep scheduler schedule, starting from a start_node, a series of walking_phases, each walking phase is composed by a series of walking_actions and each action
    stand for a certain number of node (nodes_per_action).

    The total number of nodes which will used to walk (walking_nodes) is given by:

        walking_nodes = walking_actions * nodes_per_action * walking_phases

    while the walking will start from start_node to end_node = start_node + walking_nodes - 1

    """
    def __init__(self, start_node, walking_phases, nodes_per_action, total_number_of_nodes, actions_dict):
        """
        Constructor
        Args:
            start_node: node to start to walk
            walking_phases: how many walking phases will be considered
            nodes_per_action: the number of node associated to each action
            total_number_of_nodes: total number of nodes in the optimization node
            actions_dict: dictionary of actions. Each action is formed by an ID and a list of constraint active for that action.
                Collections.OrderedDict() should be used to keep actions order!
        """
        super(footsteps_scheduler, self).__init__()

        self.__actions_dict = actions_dict

        self.__walking_actions = actions_dict.keys()

        self.__start_node = start_node
        self.__walking_phases = walking_phases
        self.__total_number_of_nodes = total_number_of_nodes
        self.__nodes_per_action = nodes_per_action

        self.__walking_nodes = len(self.__walking_actions) * self.__nodes_per_action * self.__walking_phases
        if self.__walking_nodes > total_number_of_nodes:
            FOOTSTEP_SCHEDULER_ERROR = 'walking nodes > total number of nodes : ' + str(self.__walking_nodes) + ' > ' + str(self.__total_number_of_nodes)
            raise Exception(FOOTSTEP_SCHEDULER_ERROR)

        self.__end_node = self.__start_node + self.__walking_nodes - 1
        if self.__end_node > (self.__total_number_of_nodes - 1):
            FOOTSTEP_SCHEDULER_ERROR = 'walking final node > planned horizon: ' + str(self.__end_node) + ' > ' + str(self.__total_number_of_nodes - 1)
            raise Exception(FOOTSTEP_SCHEDULER_ERROR)

        self.__actions_per_walking_phase = []
        self.__action_scheduler__()


        self.__scheduled_walking = []
        self.__walking_phases_scheduler__()

        if len(self.__scheduled_walking) != self.__walking_nodes:
            FOOTSTEP_SCHEDULER_ERROR = 'len(self.__scheduled_walking) != self.__walking_nodes: ' + str(len(self.__scheduled_walking)) + ' > ' + str(self.__walking_nodes)
            raise Exception(FOOTSTEP_SCHEDULER_ERROR)

        print 'number of nodes for walking: ', len(self.__scheduled_walking)

        print "creating constraints"
        self.__constraint_creator__()

    def __action_scheduler__(self):
        """
        The action scheduler creates a list of actions based on nodes_action
        """
        for action in enumerate(self.__walking_actions):
            for n in range(self.__nodes_per_action):
                self.__actions_per_walking_phase.append(action)

    def __walking_phases_scheduler__(self):
        """
        The walking phase scheduler creates a list of actions per each walking phase
        """
        for i in range(self.__walking_phases):
            self.__scheduled_walking += self.__actions_per_walking_phase

    def __constraint_creator__(self):
        """
        Based on the scheduled_walking it creates a proper set of constraints from the given actions_dict
        """
        sn = self.__start_node
        for action in self.__scheduled_walking:
            constraint_list = self.__actions_dict[action[1]]
            for cons in constraint_list:
                gk, g_mink, g_maxk = constraint(cons, sn, sn+1)
                self.set_constraint(gk, g_mink, g_maxk)
            sn = sn + 1

    def getStartingNode(self):
        """
        Get node from which the walking starts
        Returns:
            start node
        """
        return self.__start_node

    def getEndingNode(self):
        """
        Get node which the walking ends
        Returns:
            end node
        """
        return self.__end_node

    def getNumberOfWalkingNode(self):
        """
        Get the total number of walking nodes
        Returns:
            number of walking ndoes
        """
        return self.__walking_nodes

    def getActionsPerWalkingPhase(self):
        """
        Return the list of actions per walking phase
        Returns:
            list of actions per walking phase
            NOTE: this consider as well the number of specified nodes per action
        """
        return self.__actions_per_walking_phase

    def getNodesPerWalkingPhase(self):
        """
        Get the number of nodes a walking phase leasts
        Returns:
            number of nodes per walking phase
        """
        return len(self.__actions_per_walking_phase)

    def getScheduledWalking(self):
        """
        The final scheduled walking as a list
        Returns:
            list of shceduled walking
            NOTE: len(scheduled_walking) == getNumberOfWalkingNodes
        """
        return self.__scheduled_walking

    def printInfo(self):
        """
        Print information
        """
        print "Total walking nodes: ", self.getNumberOfWalkingNode()
        print "Starting from node ", self.getStartingNode(), " to node ", self.getEndingNode()
        print 'Actions per walking phase: ', self.getActionsPerWalkingPhase()
        print 'Number of nodes for single walking phase: ', self.getNodesPerWalkingPhase()
        print 'Scheduled walking: ', self.getScheduledWalking()

