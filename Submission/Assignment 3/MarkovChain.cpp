#include "MarkovChain.h"

/*
	Custom constructor
*/
CS397::MarkovChain::MarkovChain(const std::vector<MarkovState>& States, const std::vector<double>& transitionMat, double discountFactor)
{
	mTtransitionMat.mSize = static_cast<int>(sqrt(transitionMat.size()));
	mTtransitionMat.mValues = transitionMat;
	mOriginalTtransitionMat = mTtransitionMat;
	mDiscountFactor = discountFactor;
	mStates = States;
	mStatesValues = std::vector<double>(mStates.size(), {0});
}

/*
	Iterate once and get new state values
*/
void CS397::MarkovChain::Iteration()
{

	std::vector<double> tempStatesValues = std::vector<double>(mStates.size(), { 0 });
	
	for (int j = 0; j < mStates.size(); j++)
	{
		tempStatesValues[j] = mStates[j].mReward + mDiscountFactor * OtherStateSumation(j, mStatesValues);
			
	}

	mStatesValues = tempStatesValues;
}

/*
	Given an initial probability for each state and a defined
	number of transitions, the function will return the probability of being in each state
	after those transitions
*/
std::vector<double> CS397::MarkovChain::GetProbabilityNTransitions(const std::vector<double> initialProbabilities, unsigned numTransitions) const
{
	std::vector<double> result = initialProbabilities;

	TransitionMatrix mat = mTtransitionMat;

	for (unsigned i = 0; i < numTransitions; i++)
	{
		mat.mValues = Concatenate(mat.mValues, mTtransitionMat.mValues, mat.mSize);
	}

	std::vector<double> temp;
	for (unsigned j = 0; j < mat.mSize; j++)
	{
		double value = 0;

		for (unsigned k = 0; k < mat.mSize; k++)
		{
			value += result[k] * mat.mValues[mat.mSize * k + j];

		}

		temp.push_back(value);

	}
	result = temp;

	return result;

	
}
/*
	Returns the current state values
*/
std::vector<double> CS397::MarkovChain::GetStateValues() const
{
	return mStatesValues;
}
/*
	Sumation of the rewards of other states
*/
double CS397::MarkovChain::OtherStateSumation(const int& currentState, const std::vector<double>& values) const
{
	double value = 0;
	for (int i = 0; i < mStates.size(); i++)
	{
		value += mStatesValues[i] * mTtransitionMat.mValues[currentState * mTtransitionMat.mSize + i];
	}

	return value;
}
/*
	Multiplication between two matrices
*/
std::vector<double> CS397::MarkovChain::Concatenate(const std::vector<double> & A, const std::vector<double> & B, int size) const
{
	std::vector<double> mat = std::vector<double>(A.size(), { 0 });

	for (int i = 0; i < size; i++)
	{
		for (int col = 0; col < size; col++)
		{
			double value = 0;

			std::vector<double> A_row,B_column;
			for (int j = 0; j < size; j++)
			{
				A_row.push_back(A[i * size + j]);
				B_column.push_back(B[j * size + col]);
			}

			for (int c = 0; c < size; c++)
				value += A_row[c] * B_column[c];

			mat[i * size + col] = value;
		}
	}


	return mat;
}
