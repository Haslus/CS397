#include "MarkovDecisionProcess.h"

/*
	Custom constructor
*/
CS397::MarkovDecisionProcess::MarkovDecisionProcess(const std::vector<MarkovState>& mStates, const std::vector<MarkovAction>& mActions, double mDiscountFactor)
{
	this->mStates = mStates;
	this->mActions = mActions;
	this->mDiscountFactor = mDiscountFactor;
	this->mStateValues = std::vector<double>(mStates.size(), { 0 });
	this->mBestPolicy = std::vector<unsigned>(mStates.size(), { 0 });
}
/*
	Iterate once and calculate new state values
*/
void CS397::MarkovDecisionProcess::Iteration()
{
	std::vector<double> tempStatesValues = std::vector<double>(mStates.size(), { 0 });
	int best_action;
	for (int j = 0; j < mStates.size(); j++)
	{
		
		tempStatesValues[j] = mStates[j].mReward + mDiscountFactor * OtherStateSumation(j, best_action);
		mBestPolicy[j] = (best_action);

		if (mStates[j].mStatic)
		{
			tempStatesValues[j] = mStates[j].mReward;
		}
	}
	
	mStateValues = tempStatesValues;
}

/*
	Get current state values
*/
std::vector<double> CS397::MarkovDecisionProcess::GetStateValues() const
{
	return mStateValues;
}
/*
	Get best policy so far
*/
std::vector<unsigned> CS397::MarkovDecisionProcess::GetBestPolicy() const
{
	return mBestPolicy;
}
/*
	Helper function that acts as the sumation of the rewards of other states + the policy
*/
double CS397::MarkovDecisionProcess::OtherStateSumation(const int& currentState, int & best_action)
{
	double value = 0;
	std::vector<double> tempactions = std::vector<double>(mActions.size(), { 0 });

	for (int j = 0; j < mActions.size(); j++)
	{
		for (int i = 0; i < mStates.size(); i++)
		{
			tempactions[j] += mStateValues[i] * mActions[j].mTransitionMat.mValues[currentState * mActions[j].mTransitionMat.mSize + i];
		}
	}

	double max_value = tempactions[0];
	best_action = 0;
	for (int i = 1; i < tempactions.size(); i++)
	{
		if (tempactions[i] > max_value)
		{
			best_action = i;
			max_value = tempactions[i];
		}
	}

	return max_value;
}
