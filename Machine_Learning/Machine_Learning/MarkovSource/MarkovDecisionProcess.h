//---------------------------------------------------------------------------
#ifndef MARKOV_DECISION_PROCESS_H
#define MARKOV_DECISION_PROCESS_H
//---------------------------------------------------------------------------

#include "MarkovUtils.h"

namespace CS397
{

class MarkovDecisionProcess
{
  public:
    // constructor receives all the data to execute the algorithm
    MarkovDecisionProcess(const std::vector<MarkovState> &  mStates,  // container of the different states to analyze
                          const std::vector<MarkovAction> & mActions, // actions that can be taken (they store their respective transition matrices)
                          double                            mDiscountFactor);                    // discount factor that reduces the rewards every transition

    // iterates on the state values and best policy once
    void Iteration();

    // returns the last computed values of each state
    std::vector<double> GetStateValues() const;
    // returns the last computed best policy (action to take in each state)
    std::vector<unsigned> GetBestPolicy() const;


	//Student stuff
	double OtherStateSumation(const int& currentState, int & best_action);

	std::vector<MarkovState> mStates;
	std::vector<MarkovAction> mActions;
	double mDiscountFactor;
	std::vector<double> mStateValues;
	std::vector<unsigned> mBestPolicy;
};

} // namespace CS397

#endif