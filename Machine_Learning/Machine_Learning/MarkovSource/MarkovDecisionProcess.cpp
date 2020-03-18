#include "MarkovDecisionProcess.h"

CS397::MarkovDecisionProcess::MarkovDecisionProcess(const std::vector<MarkovState>& states, const std::vector<MarkovAction>& actions, double discountFactor)
{
	this->states = states;
	this->actions = actions;
	this->discountFactor = discountFactor;
	this->stateValues = std::vector<double>(states.size(), { 0 });
	this->bestPolicy = std::vector<unsigned>(states.size(), { 0 });
}

void CS397::MarkovDecisionProcess::Iteration()
{
	std::vector<double> tempStatesValues = std::vector<double>(states.size(), { 0 });
	int best_action;
	for (int j = 0; j < states.size(); j++)
	{
		
		tempStatesValues[j] = states[j].mReward + discountFactor * OtherStateSumation(j, best_action);
		bestPolicy[j] = (best_action);

		if (states[j].mStatic)
		{
			tempStatesValues[j] = states[j].mReward;
		}
	}
	
	stateValues = tempStatesValues;
}

std::vector<double> CS397::MarkovDecisionProcess::GetStateValues() const
{
	return stateValues;
}

std::vector<unsigned> CS397::MarkovDecisionProcess::GetBestPolicy() const
{
	return bestPolicy;
}

double CS397::MarkovDecisionProcess::OtherStateSumation(const int& currentState, int & best_action)
{
	double value = 0;
	std::vector<double> tempactions = std::vector<double>(actions.size(), { 0 });

	for (int j = 0; j < actions.size(); j++)
	{
		for (int i = 0; i < states.size(); i++)
		{
			tempactions[j] += stateValues[i] * actions[j].mTransitionMat.mValues[currentState * actions[j].mTransitionMat.mSize + i];
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
