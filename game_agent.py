"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import math

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # Get Opponent
    opponent = game.get_opponent(player)

    # If there is already a winner
    if game.is_winner(player):
        return float('inf')
    elif game.is_winner(opponent):
        return float('-inf')

    return custom_score_award_edge_moves_magnify(game, player)


def custom_score_my_vs_op_moves(game, player):

    """
    ID Improved: 70.71%
    Student: 62.14%

    This is the default heuristic from lecture i use it here as a benchmark.
    """

    # Get Opponent
    opponent = game.get_opponent(player)

    # Evaluation function
    number_my_moves = len(game.get_legal_moves(player))
    number_op_moves = len(game.get_legal_moves(opponent))

    return float(number_my_moves - number_op_moves)


def custom_score_award_edge_moves(game, player):

    """
    ID Improved: 69.29%
    Student: 70.71%

    This is heuristic since the moves is similar to the knight moves in chess i used the same idea in
    Warnsdorff’s algorithm to award edge moves and penalize moves near to center as this will require
    the agent to search very far, as the edge moves will have less next moves the agent will be done with all edge
    moves first and then explore moves near the center if there is enough time.

    I used 50 as constant to penalize the bad moves more.
    """

    # Get Opponent
    opponent = game.get_opponent(player)

    # Get locations
    my_location = game.get_player_location(player)
    op_location = game.get_player_location(opponent)

    # Determine how far from center
    center_x = int(game.width / 2)
    center_y = int(game.height / 2)

    my_steps_from_center = abs(my_location[0] - center_x) + abs(my_location[1] - center_y)
    op_steps_from_center = abs(op_location[0] - center_x) + abs(op_location[1] - center_y)

    # Available number of moves
    my_moves = len(game.get_legal_moves(player))
    op_moves = len(game.get_legal_moves(opponent))

    return float((my_moves + my_steps_from_center) - 100 * (op_moves + op_steps_from_center))


def custom_score_award_edge_moves_distance(game, player):

    """
    ID Improved: 67.86%
    Student: 59.29%
    """

    # Get Opponent
    opponent = game.get_opponent(player)

    # Get locations
    my_location = game.get_player_location(player)
    op_location = game.get_player_location(opponent)

    # Determine how far from center
    center_x = int(game.width / 2)
    center_y = int(game.height / 2)

    my_steps_from_center = abs(my_location[0] - center_x) + abs(my_location[1] - center_y)
    op_steps_from_center = abs(op_location[0] - center_x) + abs(op_location[1] - center_y)

    # Distance from center

    my_distance = math.sqrt((center_x - my_location[0]) ** 2 + (center_y - my_location[1]) ** 2)
    op_distance = math.sqrt((center_x - op_location[0]) ** 2 + (center_y - op_location[1]) ** 2)

    return float(my_distance - 100 * op_distance)


def custom_score_award_edge_moves_magnify(game, player):
    """
    ID Improved: 71.43%
    Student: 71.14%

    This is just and experimental heuristic that uses the same idea above however here i change the equation by
    multiplying how far player from center is with the number of moves which will magnify the scores.
    """
    # Get Opponent
    opponent = game.get_opponent(player)

    # Get locations
    my_location = game.get_player_location(player)
    op_location = game.get_player_location(opponent)

    # Determine how far from center
    center_x = int(game.width / 2)
    center_y = int(game.height / 2)

    my_steps_from_center = abs(my_location[0] - center_x) + abs(my_location[1] - center_y)
    op_steps_from_center = abs(op_location[0] - center_x) + abs(op_location[1] - center_y)

    # Available number of moves
    my_moves = len(game.get_legal_moves(player))
    op_moves = len(game.get_legal_moves(opponent))

    return float((my_moves * my_steps_from_center) - 25 * (op_moves * op_steps_from_center))


def custom_score_blank_spaces(game, player):
    """
    ID Improved: 63.57%
    Student: 73.57%

    In this heuristic i used the number of remaining squares to motivate selecting good moves especially at the
    beginning of the game.
    """
    # Get Opponent
    opponent = game.get_opponent(player)

    blank_spaces = len(game.get_blank_spaces())

    # Available number of moves
    my_moves = len(game.get_legal_moves(player))
    op_moves = len(game.get_legal_moves(opponent))

    return float((my_moves - blank_spaces) - (op_moves * blank_spaces))



class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        best_score = float('-inf')
        best_move = (-1, -1)

        if not legal_moves:
            return best_move

        center_x = int(game.width / 2)
        center_y = int(game.height / 2)
        center_move = (center_x, center_y)

        # Set of moves other player cannot reflect
        opening_book = [(center_x - 1, center_y),(center_x, center_y - 1),(center_x + 1, center_y),
                        (center_x, center_y + 1),
                        (center_x - 1, center_y - 1), (center_x + 1, center_y + 1), (center_x + 1, center_y - 1),
                        (center_x - 1, center_y + 1)]

        if game.move_count == 0:
            if center_move in legal_moves:
                return center_move
            else:
                return opening_book[random.randint(0, len(opening_book)-1)]
        elif game.move_count == 1:
            if center_move in legal_moves:
                return center_move
            else:
                return opening_book[random.randint(0, len(opening_book)-1)]

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring

            if self.iterative:

                current_depth = 1

                while True:

                    if self.method == 'alphabeta':
                        _, best_move = self.alphabeta(game, current_depth)
                    else:
                        _, best_move = self.minimax(game, current_depth)

                    current_depth += 1
                else:
                    raise Timeout()

            else:
                if self.method == 'alphabeta':
                    _, best_move = self.alphabeta(game, self.search_depth)
                else:
                    _, best_move = self.minimax(game, self.search_depth)

        except Timeout:
            # Handle any actions required at timeout, if necessary
            return best_move

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # Base Case: if there is no possible moves or the depth is 0
        if depth <= 0:
            return self.score(game, self), (-1, -1)

        # Get possible moves
        legal_moves = game.get_legal_moves()

        if not legal_moves:
            return self.score(game, self), (-1, -1)

        score_move_list = []
        for current_move in legal_moves:
            current_game = game.forecast_move(current_move)

            current_score, _ = self.minimax(current_game, depth-1, not maximizing_player)

            score_move_list.append((current_score, current_move))

        if maximizing_player:
            return max(score_move_list)
        else:
            return min(score_move_list)


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # Base Case: if there is no possible moves or the depth is 0
        if depth <= 0:
            return self.score(game, self), (-1, -1)

        # Get possible moves
        legal_moves = game.get_legal_moves()

        if not legal_moves:
            return self.score(game, self), (-1, -1)

        if maximizing_player:
            best_value = float('-inf')
            best_move = (-1, -1)

            for current_move in legal_moves:
                current_game = game.forecast_move(current_move)

                current_score, _ = self.alphabeta(current_game, depth-1, alpha, beta, not maximizing_player)

                if current_score > best_value:
                    best_value = current_score
                    best_move = current_move
                if best_value >= beta:
                    return best_value, best_move
                alpha = max(best_value, alpha)
        else:
            best_value = float('inf')
            best_move = (-1, -1)

            for current_move in legal_moves:
                current_game = game.forecast_move(current_move)

                current_score, _ = self.alphabeta(current_game, depth - 1, alpha, beta, not maximizing_player)

                if current_score < best_value:
                    best_value = current_score
                    best_move = current_move
                if best_value <= alpha:
                    return best_value, best_move
                beta = min(best_value, beta)

        return best_value, best_move
