This is the result of applying MarkovDecisionProcess to the GridWorld10x10

So from this 
  0   0   0   0   0   0   0   0   0   0
  0   X   0   0   0   0   0   0   0   0
  0   X   1   X   0   0   0   0   0   0
  0   0   0   X   0   0   0   0   0   0
-10 -10 -10   0   0   0   0   0   0   0
100   X   0   0   0   0   0   0   0   0
  0   X   0   0   0   0   0   0   0   0
  0   X   0   0   0 -10   0   0   0   0
  0   0   0   0   0 -10   0   0   0   0
  0   0   0   0   0 -10   0   0   0   0
  
to this

1 1 1 1 2 2 3 3 3 2
0 X 1 1 2 2 2 3 3 2
0 X 0 X 2 2 2 3 2 2
0 3 3 X 2 2 3 3 3 2
2 0 2 2 2 3 3 3 3 3
2 X 2 2 2 3 3 3 3 3
0 X 2 2 3 3 3 3 3 3
0 X 2 3 3 3 0 0 3 3
0 3 3 3 3 3 0 0 0 0
0 3 3 3 3 3 0 0 0 0

Each number represents an action.
0 : Up
1 : Right
2 : Down
3 : Left

So what we can interpret from this result is that each tile has the best action (or movement) to reach the best tile
which would be 100 on the original grid. For example, if you start on the bottom right corner, you will see that your
actions will try to go left and over the column of -10, and then try to reach the 100. You can see that, no matter in which tile
you start, you will end up in 100, while also avoiding the -10 ones.