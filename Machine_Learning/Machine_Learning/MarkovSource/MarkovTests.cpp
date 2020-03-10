
#include "gtest/gtest.h"

#include "MarkovChain.h"
#include "MarkovDecisionProcess.h"

#include <array>
#include <fstream>
#include <iostream>
#include <unordered_map>

using namespace CS397;

template <typename T>
void AssertVectors(const std::vector<T> & val1, const std::vector<T> & val2)
{
    ASSERT_EQ(val1.size(), val2.size());

    for (size_t i = 0; i < val1.size(); i++)
    {
        ASSERT_NEAR(val1[i], val2[i], 0.01);
    }
}

TEST(MarkovChain, Init)
{
    MarkovChain chain = MarkovChain(
        { { "State1", 2.0 },
          { "State2", 5.0 },
          { "State3", 10.0 } },
        { 0.50, 0.50, 0.00,
          0.25, 0.50, 0.25,
          0.00, 0.34, 0.66 },
        0.9);

    AssertVectors<double>({ 0.0, 0.0, 0.0 }, chain.GetStateValues());
}

TEST(MarkovChain, ProbabiltyNTransitions)
{
    MarkovChain chain = MarkovChain(
        { { "State1", 2.0 },
          { "State2", 5.0 },
          { "State3", 10.0 } },
        { 0.50, 0.50, 0.00,
          0.25, 0.50, 0.25,
          0.00, 0.34, 0.66 },
        0.9);

    std::vector<double> probTransitions = chain.GetProbabilityNTransitions({ 1.0, 0.0, 0.0 }, 3);

    AssertVectors<double>({ 0.27, 0.47, 0.26 }, probTransitions);

    probTransitions = chain.GetProbabilityNTransitions({ 1.0, 0.0, 0.0 }, 30);

    AssertVectors<double>({ 0.22, 0.44, 0.33 }, probTransitions);
}

TEST(MarkovChain, IterateStateFeel_DF09)
{
    MarkovChain chain = MarkovChain(
        { { "Sad", 10.0 },
          { "Neutral", 1.0 },
          { "Happy", 8.0 } },
        { 0.20, 0.80, 0.00,
          0.10, 0.20, 0.70,
          0.10, 0.40, 0.5 },
        0.9);

    for (size_t i = 0; i < 100; i++)
    {
        chain.Iteration();
    }
    AssertVectors<double>({ 58.36, 52.58, 58.51 }, chain.GetStateValues());
}

TEST(MarkovChain, IterateState100_DF09)
{
    MarkovChain chain = MarkovChain(
        { { "State1", 2.0 },
          { "State2", 5.0 },
          { "State3", 10.0 } },
        { 0.50, 0.50, 0.00,
          0.25, 0.50, 0.25,
          0.00, 0.34, 0.66 },
        0.9);

    for (size_t i = 0; i < 100; i++)
    {
        chain.Iteration();
    }
    AssertVectors<double>({ 50.98, 57.86, 68.24 }, chain.GetStateValues());
}

TEST(MarkovChain, IterateState200_DF07)
{
    MarkovChain chain = MarkovChain(
        { { "State1", 2.0 },
          { "State2", 5.0 },
          { "State3", 10.0 } },
        { 0.50, 0.50, 0.00,
          0.25, 0.50, 0.25,
          0.00, 0.34, 0.66 },
        0.7);

    for (size_t i = 0; i < 10; i++)
    {
        chain.Iteration();
    }
    AssertVectors<double>({ 12.41, 17.82, 26.15 }, chain.GetStateValues());

    for (size_t i = 0; i < 190; i++)
    {
        chain.Iteration();
    }
    AssertVectors<double>({ 12.97, 18.38, 26.72 }, chain.GetStateValues());
}

TEST(MarkovDecisionProcess, Iteration100_DF09)
{
    MarkovDecisionProcess mdp = MarkovDecisionProcess(
        { { "Sad", 10.0 },
          { "Neutral", 1.0 },
          { "Happy", 8.0 } },
        { { "Cry",
            { 3,
              { 0.20, 0.70, 0.10,
                0.50, 0.30, 0.20,
                0.10, 0.70, 0.20 } } },
          { "Laugh",
            { 3,
              { 0.00, 0.20, 0.80,
                0.10, 0.40, 0.50,
                0.10, 0.20, 0.70 } } } },
        0.9);

    for (size_t i = 0; i < 100; i++)
    {
        mdp.Iteration();
    }

    AssertVectors<double>({ 70.94, 62.14, 69.11 }, mdp.GetStateValues());

    AssertVectors<unsigned>({ 1u, 0u, 1u }, mdp.GetBestPolicy());
}

TEST(MarkovDecisionProcess, GridWorld_DF099_R003)
{
    const double Reward = -0.03;

    MarkovDecisionProcess mdp = MarkovDecisionProcess(
        { { "0", Reward },
          { "1", Reward },
          { "2", Reward },
          { "3", 1, true },
          { "4", Reward },
          //{ "5", Reward },
          { "6", Reward },
          { "7", -1, true },
          { "8", Reward },
          { "9", Reward },
          { "10", Reward },
          { "11", Reward } },
        /*
		-------------
		| 0| 1| 2| 3|
		-------------
		| 4| X| 6| 7|
		-------------
		| 8| 9|10|11|
		-------------
		*/
        { { "Up",
            { 11,
              //  0    1    2    3    4    6    7    8    9   10   11
              { 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.8, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.8, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.1, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.0, 0.1,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.1 } } },
          { "Right",
            { 11,
              //  0    1    2    3    4    6    7    8    9   10   11
              { 0.1, 0.8, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.2, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.1, 0.8, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0,
                0.1, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.0,
                0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.1,
                0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.8, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.8, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.8,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.9 } } },
          { "Down",
            { 11,
              //  0    1    2    3    4    6    7    8    9   10   11
              { 0.1, 0.1, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.1, 0.0, 0.1, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.8, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.8,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.1, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.1,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.9 } } },
          { "Left",
            { 11,
              //  0    1    2    3    4    6    7    8    9   10   11
              { 0.9, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.8, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.8, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0,
                0.1, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.1, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.1, 0.0,
                0.0, 0.0, 0.0, 0.1, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.1,
                0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.8, 0.1, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.8, 0.1 } } } },
        0.99);

    for (size_t i = 0; i < 100; i++)
    {
        mdp.Iteration();
    }

    AssertVectors<double>({ 0.82, 0.87, 0.92, 1.00, 0.77, 0.66, -1.00, 0.72, 0.67, 0.63, 0.41 }, mdp.GetStateValues());

    AssertVectors<unsigned>({ 1, 1, 1, 0, 0, 0, 0, 0, 3, 3, 3 }, mdp.GetBestPolicy());
}

TEST(MarkovDecisionProcess, GridWorld_DF099_R005)
{
    const double Reward = -0.05;

    MarkovDecisionProcess mdp = MarkovDecisionProcess(
        { { "0", Reward },
          { "1", Reward },
          { "2", Reward },
          { "3", 1, true },
          { "4", Reward },
          //{ "5", Reward },
          { "6", Reward },
          { "7", -1, true },
          { "8", Reward },
          { "9", Reward },
          { "10", Reward },
          { "11", Reward } },
        /*
		-------------
		| 0| 1| 2| 3|
		-------------
		| 4| X| 6| 7|
		-------------
		| 8| 9|10|11|
		-------------
		*/
        { { "Up",
            { 11,
              //  0    1    2    3    4    6    7    8    9   10   11
              { 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.8, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.8, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.1, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.0, 0.1,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.1 } } },
          { "Right",
            { 11,
              //  0    1    2    3    4    6    7    8    9   10   11
              { 0.1, 0.8, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.2, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.1, 0.8, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0,
                0.1, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.0,
                0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.1,
                0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.8, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.8, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.8,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.9 } } },
          { "Down",
            { 11,
              //  0    1    2    3    4    6    7    8    9   10   11
              { 0.1, 0.1, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.1, 0.0, 0.1, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.8, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.8,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.1, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.1,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.9 } } },
          { "Left",
            { 11,
              //  0    1    2    3    4    6    7    8    9   10   11
              { 0.9, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.8, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.8, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0,
                0.1, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.1, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.1, 0.0,
                0.0, 0.0, 0.0, 0.1, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.1,
                0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.8, 0.1, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.8, 0.1 } } } },
        0.99);

    for (size_t i = 0; i < 100; i++)
    {
        mdp.Iteration();
    }

    AssertVectors<double>({ 0.74, 0.82, 0.89, 1.00, 0.67, 0.62, -1.00, 0.59, 0.52, 0.52, 0.29 }, mdp.GetStateValues());

    AssertVectors<unsigned>({ 1, 1, 1, 0, 0, 0, 0, 0, 3, 0, 3 }, mdp.GetBestPolicy());
}

TEST(MarkovDecisionProcess, GridWorld_DF099_R2)
{
    const double Reward = -2.0;

    MarkovDecisionProcess mdp = MarkovDecisionProcess(
        { { "0", Reward },
          { "1", Reward },
          { "2", Reward },
          { "3", 1, true },
          { "4", Reward },
          //{ "5", Reward },
          { "6", Reward },
          { "7", -1, true },
          { "8", Reward },
          { "9", Reward },
          { "10", Reward },
          { "11", Reward } },
        /*
		-------------
		| 0| 1| 2| 3|
		-------------
		| 4| X| 6| 7|
		-------------
		| 8| 9|10|11|
		-------------
		*/
        { { "Up",
            { 11,
              //  0    1    2    3    4    6    7    8    9   10   11
              { 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.8, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.8, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.1, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.0, 0.1,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.1 } } },
          { "Right",
            { 11,
              //  0    1    2    3    4    6    7    8    9   10   11
              { 0.1, 0.8, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.2, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.1, 0.8, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0,
                0.1, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.0,
                0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.1,
                0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.8, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.8, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.8,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.9 } } },
          { "Down",
            { 11,
              //  0    1    2    3    4    6    7    8    9   10   11
              { 0.1, 0.1, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.1, 0.0, 0.1, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.8, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.8,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.1, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.1,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.9 } } },
          { "Left",
            { 11,
              //  0    1    2    3    4    6    7    8    9   10   11
              { 0.9, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.8, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.8, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0,
                0.1, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.1, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.1, 0.0,
                0.0, 0.0, 0.0, 0.1, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.1,
                0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.8, 0.1, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.8, 0.1 } } } },
        0.99);

    for (size_t i = 0; i < 100; i++)
    {
        mdp.Iteration();
    }

    AssertVectors<double>({ -6.94, -4.20, -1.73, 1.00, -9.35, -3.55, -1.00, -10.56, -8.32, -5.90, -3.75 }, mdp.GetStateValues());

    AssertVectors<unsigned>({ 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0 }, mdp.GetBestPolicy());
}

void LoadGridWorld(const std::string & filename, double reward, std::vector<MarkovState> & states, std::vector<MarkovAction> & actions)
{
    std::ifstream inFile(filename);

    if (!inFile.is_open())
        return;

    int index    = 0;
    int cellName = 0;

    std::unordered_map<int, int> stateToIndex;

    int width;
    int height;

    // read grid size
    inFile >> width;
    inFile >> height;

    // read all values per cell
    while (!inFile.eof())
    {
        std::string value;
        inFile >> value;

        if (value.size() == 0)
            continue;

        if (value == "0")
        {
            states.push_back({ std::to_string(cellName), reward });
            stateToIndex[cellName] = index++;
        }
        else if (value != "X")
        {
            states.push_back({ std::to_string(cellName), std::atof(value.c_str()), true });
            stateToIndex[cellName] = index++;
        }
        cellName++;
    }

    // create transition matrices for each action
    std::vector<double> upTransition(states.size() * states.size(), 0.0);
    std::vector<double> rightTransition(states.size() * states.size(), 0.0);
    std::vector<double> downTransition(states.size() * states.size(), 0.0);
    std::vector<double> leftTransition(states.size() * states.size(), 0.0);

    // fill the matrices
    for (unsigned i = 0; i < states.size(); i++)
    {
        int cellNameIndex = std::atoi(states[i].mName.c_str());
        int cellIndex     = stateToIndex[cellNameIndex];

        // self transition is always possible
        upTransition[cellIndex * states.size() + cellIndex]    = 1.0;
        rightTransition[cellIndex * states.size() + cellIndex] = 1.0;
        downTransition[cellIndex * states.size() + cellIndex]  = 1.0;
        leftTransition[cellIndex * states.size() + cellIndex]  = 1.0;

        // set transtions for up action (move up can lead to up/left/right)
        int upIdx = cellNameIndex - width;
        if (stateToIndex.find(upIdx) != stateToIndex.end())
        {
            upTransition[i * states.size() + stateToIndex[upIdx]] = 8.0;

            rightTransition[i * states.size() + stateToIndex[upIdx]] = 0.1;
            leftTransition[i * states.size() + stateToIndex[upIdx]]  = 0.1;
        }

        // set transtions for right action (move up can lead to right/down/up)
        int rightIdx = (cellNameIndex + 1) % width == 0 ? -1 : cellNameIndex + 1;
        if (stateToIndex.find(rightIdx) != stateToIndex.end())
        {
            rightTransition[i * states.size() + stateToIndex[rightIdx]] = 8.0;

            upTransition[i * states.size() + stateToIndex[rightIdx]]   = 0.1;
            downTransition[i * states.size() + stateToIndex[rightIdx]] = 0.1;
        }

        // set transtions for down action (move up can lead to down/right/left)
        int downIdx = (cellNameIndex + width) > width * height ? -1 : cellNameIndex + width;
        if (stateToIndex.find(downIdx) != stateToIndex.end())
        {
            downTransition[i * states.size() + stateToIndex[downIdx]] = 8.0;

            rightTransition[i * states.size() + stateToIndex[downIdx]] = 0.1;
            leftTransition[i * states.size() + stateToIndex[downIdx]]  = 0.1;
        }

        // set transtions for left action (move up can lead to left/up/down)
        int leftIdx = cellNameIndex % width == 0 ? -1 : cellNameIndex - 1;
        if (stateToIndex.find(leftIdx) != stateToIndex.end())
        {
            leftTransition[i * states.size() + stateToIndex[leftIdx]] = 8.0;

            upTransition[i * states.size() + stateToIndex[leftIdx]]   = 0.1;
            downTransition[i * states.size() + stateToIndex[leftIdx]] = 0.1;
        }
    }

    // normalize transitions so that all probabilities add up to one
    for (unsigned i = 0; i < states.size(); i++)
    {
        double sumUp    = 0.0;
        double sumRight = 0.0;
        double sumDown  = 0.0;
        double sumLeft  = 0.0;

        for (unsigned j = 0; j < states.size(); j++)
        {
            sumUp += upTransition[i * states.size() + j];
            sumRight += rightTransition[i * states.size() + j];
            sumDown += downTransition[i * states.size() + j];
            sumLeft += leftTransition[i * states.size() + j];
        }

        for (unsigned j = 0; j < states.size(); j++)
        {
            if (sumUp != 0.0)
                upTransition[i * states.size() + j] /= sumUp;
            if (sumRight != 0.0)
                rightTransition[i * states.size() + j] /= sumRight;
            if (sumDown != 0.0)
                downTransition[i * states.size() + j] /= sumDown;
            if (sumLeft != 0.0)
                leftTransition[i * states.size() + j] /= sumLeft;
        }
    }

    // add actions to list
    actions.push_back({ "Up", { static_cast<unsigned>(states.size()), upTransition } });
    actions.push_back({ "Right", { static_cast<unsigned>(states.size()), rightTransition } });
    actions.push_back({ "Down", { static_cast<unsigned>(states.size()), downTransition } });
    actions.push_back({ "Left", { static_cast<unsigned>(states.size()), leftTransition } });
}

TEST(MarkovDecisionProcess, GridWorld01_DF09_R001)
{
    const double Reward = -0.01;

    std::vector<MarkovState>  states;
    std::vector<MarkovAction> actions;

    LoadGridWorld("GridWorld01.txt", Reward, states, actions);

    MarkovDecisionProcess mdp = MarkovDecisionProcess(states, actions, 0.9);

    for (size_t i = 0; i < 100; i++)
    {
        mdp.Iteration();
    }

    AssertVectors<unsigned>({ 1, 1, 1, 0, 0, 3, 0, 0, 3, 3, 2 }, mdp.GetBestPolicy());
}

TEST(MarkovDecisionProcess, GridWorld10x10_DF09R001)
{
    const double Reward = -0.01;

    std::vector<MarkovState>  states;
    std::vector<MarkovAction> actions;

    LoadGridWorld("GridWorld10x10.txt", Reward, states, actions);

    MarkovDecisionProcess mdp = MarkovDecisionProcess(states, actions, 0.98);

    for (size_t i = 0; i < 100; i++)
    {
        mdp.Iteration();
    }

    std::vector<unsigned> policy = mdp.GetBestPolicy();

	// print policy in grid shape
    int cellIndex = 0;
    for (size_t i = 0; i < policy.size(); i++)
    {
        if (cellIndex % 10 == 0)
            std::cout << std::endl;

        int cellName = std::atoi(states[i].mName.c_str());

        for (int j = cellIndex; j < (cellName); j++)
        {
            std::cout << "X ";
            cellIndex++;
        }

        std::cout << policy[i] << " ";
        cellIndex++;
    }
    std::cout << std::endl
              << std::endl;
}

int main(int argc, char ** argv)
{
    testing::InitGoogleTest(&argc, argv);

    int toReturn = RUN_ALL_TESTS();

    std::system("pause");

    return toReturn;
}