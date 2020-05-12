from horizon import *

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

        self.actions_dict = actions_dict

        self.walking_actions = actions_dict.keys()
        print 'walking actions', self.walking_actions

        self.start_node = start_node
        self.walking_phases = walking_phases
        self.total_number_of_nodes = total_number_of_nodes
        self.nodes_per_action = nodes_per_action

        self.walking_nodes = len(self.walking_actions) * self.nodes_per_action * self.walking_phases
        if self.walking_nodes > total_number_of_nodes:
            FOOTSTEP_SCHEDULER_ERROR = 'walking nodes > total number of nodes : ' + str(self.walking_nodes) + ' > ' + str(self.total_number_of_nodes)
            raise Exception(FOOTSTEP_SCHEDULER_ERROR)
        print 'walking nodes: ', self.walking_nodes

        self.end_node = self.start_node + self.walking_nodes - 1
        if self.end_node > (self.total_number_of_nodes - 1):
            FOOTSTEP_SCHEDULER_ERROR = 'walking final node > planned horizon: ' + str(self.end_node) + ' > ' + str(self.total_number_of_nodes - 1)
            raise Exception(FOOTSTEP_SCHEDULER_ERROR)
        print 'walking end node:', self.end_node

        self.actions_per_walking_phase = []
        self.action_scheduler()
        print 'actions per walking phase: ', self.actions_per_walking_phase
        print 'number of nodes for single walking phase: ', len(self.actions_per_walking_phase)

        self.scheduled_walking = []
        self.walking_phases_scheduler()
        print 'scheduled walking: ', self.scheduled_walking
        print 'number of nodes for walking: ', len(self.scheduled_walking)

        print "creating constraints"
        self.constraint_creator()

    def action_scheduler(self):
        """
        The action scheduler creates a list of actions based on nodes_action
        """
        for action in enumerate(self.walking_actions):
            for n in range(self.nodes_per_action):
                self.actions_per_walking_phase.append(action)

    def walking_phases_scheduler(self):
        """
        The walking phase scheduler creates a list of actions per each walking phase
        """
        for i in range(self.walking_phases):
            self.scheduled_walking += self.actions_per_walking_phase

    def constraint_creator(self):
        """
        Based on the scheduled_walking it creates a proper set of constraints from the given actions_dict
        """
        sn = self.start_node
        for action in self.scheduled_walking:
            constraint_list = self.actions_dict[action[1]]
            print "there are ", len(constraint_list), " constraints for action ", action[1]
            for cons in constraint_list:
                gk, g_mink, g_maxk = constraint(cons, sn, sn+1)
                self.set_constraint(gk, g_mink, g_maxk)
            sn = sn + 1
