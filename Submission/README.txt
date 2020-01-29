o What exponent value to use for each feature? What are the theta values for each feature?
The features are:
X0 - Intercept, which is 1
X1 - Transaction date
X2 - House age 
X3 - Distance nearest MRT station
X4 - Number of stores
X5 - Latitude
X6 - Longitude
All of this data is already normalized. The exponent used for the first model is 1 for ALL features.
Theta is 0.0 by default.
The theta values that I get are for each feature are:
X0 - 37.98
X1 - 4.6
X2 - -11.73 
X3 - -22.05
X4 - 12.55
X5 - 17.87
X6 - 6.38


I have also tried using other exponents, combining linear, quadratic and both at the same time.
Combinations like 1/1/1/2/1/2/2 or 1/1/1/2/2/1/1/1 have not yieled better results, and by that I mean that the cost was always higher.
Using preset thetas for some values also did not improve the model, and some did improve it, but the function made no sense, at least for me,
so I discarded then.

o Write down the function the model uses to predict. Explain how you interpret the function you got.
If we write the previous data as a function, we get:
37.98 + 4.6*X1 + -11.73*X2 + -22.05*X3 + 12.55*X4 + 17.87*X5 + 6.38 * X6
Now let's interpret this.
For the first parameter (37.98), I believe it can be considered the base of the house price of unit area.
For the second parameter (4.6), it represents the transaction date. Looking at the data, it does not seem to weigh too much in the final result,
but if you visualize it, you can see the output reaches a bigger maximum the bigger the transaction date is.
For the third parameter (-11.73), represents the house age. From a logical standpoint, it makes sense that the older a house is, the less value it has,
which is why its negative.
For the fourth parameter (-22.05), it's the distance nearest MRT station. This one is simple: the farther you are from a MRT station, the worse it is for you,
sice you need to walk, which devalues the final price. I personally think this shouldn't impact so much the output, which led me to believe that the data might 
be biased against it.
For the fifth parameter, it's the number of stores near the house. The more stores you have, the more confortable it's for the client
which increases the value.
For the last two (17.87 and 6.38), I put them together since they both represent position, although one has more weigth than the other.
This might happen because some of houses from the data are along the same Latitude, but changes the Longitude.


o How many iterations does it need to converge?
It needs about 35 iterations to reach the lowest cost, 38.5659

o Why are you using the learning rate you are using?
I use 1 as the learning rate so that it converges faster. If I use 0.5, it also converges, but takes longer. If I go higher, the cost is too high, and
if use 3 or above it no longer converges.

o What if you divide the dataset into training and test sets? What is the result in the test set? Overfitting? Underfitting?
If I divide the dataset into training and test set on a 70:30 ratio, it actually gets a lower cost (33.52), which means that this model
would might be better than the first one. There are some big differences though, the big one being the latitude and longitude thetas.
They are now (21.55 and -0.94) respectly, which is pretty different from the previous, which may indeicate that the dataset may be biased,
so it may not work for fututre data.

