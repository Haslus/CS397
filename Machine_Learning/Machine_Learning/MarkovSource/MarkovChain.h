//---------------------------------------------------------------------------
#ifndef MARKOV_CHAIN_H
#define MARKOV_CHAIN_H
//---------------------------------------------------------------------------

#include "MarkovUtils.h"

namespace CS397
{

class MarkovChain
{
  public:
    // constructor receives all the data to execute the algorithm
    MarkovChain(const std::vector<MarkovState> & states,        // container of the different states to analyze
                const std::vector<double> &      transitionMat, // transition matrix with probabilities of transitioning between states
                double                           discountFactor);                         // discount factor that reduces the rewards every transition

	// iterates on the state values once
    void                Iteration();
	
	// computes the probability of being in each state after the amount of transitions specified in the parameter (numTransitions)
	// and given an initial probability to each state
    std::vector<double> GetProbabilityNTransitions(const std::vector<double> initialProbabilities, unsigned numTransitions) const;

    // returns the last computed values of each state
    std::vector<double> GetStateValues() const;

};

} // namespace CS397

#endif