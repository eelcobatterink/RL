class Agent( object ):

    def getAction(self, state, action_space):
        raise Exception( 'Not Implemented' )

    def feedback(self, nextState, reward):
        raise Exception('Not Implemented')


class RandomAgent( Agent ):
    def getAction(self, state, action_space):
        return action_space.sample()

    def feedback(self, nextState, reward):
        pass