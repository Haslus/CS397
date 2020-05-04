My first try with the final test was using a topology of 784,100,10
and a learning rate of 0.1. There wasn't a specific reason to use 100, I just
went it.

The initial accuracy increased a lot in the first iteration, but then it went down.
It didn't even reach a 70% of accuracy. I think it was because the learning rate was high
so it probably went over the result.

So for my next try, I used a learning rate of 0.01 and a topology of 784,100,30,10.
Same as before I just randomly put a smaller number than the previous one.
The first iteration I got a 73% accuracy, and by the third one I was over 80% alreeady.
So I think this topology is good enough. You could probably improve it, and have at 80%
by the first iteration.