#include "evolution.hpp"

#include <algorithm>
#include <cassert>

#include <iostream>

namespace NeuroCar {

template <typename DNAType>
Population<DNAType> evolution(Population<DNAType> const & population)
{
    #if 0
    static_assert(
        std::is_base_of<DNA<DNAType::Subject, DNAType>>::value,
        "DNAType must inherits from DNA"
    );
    #endif

    (void) population;

    assert(population.size() > 0);

    // TODO
    // Train - simulate world
#if 0
    for(auto const & dna: population)
    {
        train(dna);
    }
#endif


    // Compute the fitness of the population

    using Fitness = typename DNAType::Fitness;

    Fitness cumulativeFitness = 0.0;
    for(auto const & dna: population)
    {
        cumulativeFitness += dna.fitness();
    }

    Fitness avgFitness = cumulativeFitness / static_cast<Fitness>(population.size());

    //std::cout << "Average fitness = " << avgFitness << std::endl;


    // Create the mating pool

    //TODO: create a struct for RankedDNA -> better readability
    // + use std::ref<DNA> ?
    using RankedDNA  = std::pair<DNAType, Fitness>;
    using MatingPool = std::vector<RankedDNA>;

    MatingPool matingPool;
    matingPool.reserve(population.size());

    for(auto const & dna: population)
    {
        //TODO: cache dna fitness
        matingPool.push_back(std::make_pair(dna, dna.fitness() / cumulativeFitness));
    }

    // Reverse sort: greatest relative fitness to lowest
    std::sort(std::begin(matingPool), std::end(matingPool),
        [](RankedDNA const & lhs, RankedDNA const & rhs)
        {
            return lhs.second > rhs.second;
        }
    );

    // FIXME: find better name
    Fitness cumulativeScore = 0.0;
    for(auto & dna: matingPool)
    {
        cumulativeScore += dna.second;
        dna.second = cumulativeScore;
    }

    // Reproduce
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<Fitness> random(0.0, 1.0);

    static auto const selectParent = [&matingPool, &random, &rng]() -> DNAType const &
    {
        Fitness r = random(rng);
        for(auto i = 0u; i < matingPool.size()-1; ++i)
        {
            if(matingPool[i].second > r)
            {
                return matingPool[i].first;
            }
        }

        return matingPool[matingPool.size()-1].first;
    };

    typename DNAType::MutationRate const mutationRate = 0.01;

    // FIXME: Create a new generation at each evolutionary step or do it in-place?
    Population<DNAType> nextGeneration;
    nextGeneration.reserve(population.size());

    for(auto i = 0u; i < population.size(); ++i)
    {
        DNAType const & parentA = selectParent();
        DNAType const & parentB = selectParent();

        //TODO:
        // - When to remove the T entity?
        auto child = parentA.crossover(parentB);
        DNAType childDNA(child);
        childDNA.mutate(mutationRate);

        // TODO:
        // - How to set target of DNA? -> for the fitness
#if 0
        childDNA.setTarget(parentA.getTarget());
#endif

        nextGeneration.emplace_back(std::move(childDNA));
    }

    return nextGeneration;
}

template <typename DNAType, typename T>
Population<DNAType> simulate(Individuals<T> const & individuals, std::size_t ngenerations)
{
    Population<DNAType> population;
    population.reserve(individuals.size());

    // Create initial DNA
    for(auto i = 0u; i < individuals.size(); ++i)
    {
        assert(individuals[i] != nullptr);
        DNAType dna(individuals[i]);
        dna.randomize();
        population.emplace_back(std::move(dna));
    }

    assert(population.size() == individuals.size());

    // Evolve
    for(auto i = 0u; i < ngenerations; ++i)
    {
        std::cout << "Generation " << i << std::endl;
        population = evolution(population);
    }

    return population;
}

template <typename DNAType>
void evaluate(Population<DNAType> & population)
{

#if 0
    //TODO: create a struct for RankedDNA -> better readability
    // + use std::ref<DNA> ?
    using RankedDNA  = std::pair<DNA<T, DNAType>, Fitness>;
    using RankedDNAs = std::vector<RankedDNA>;

    RankedDNAs ranked;
    ranked.reserve(population.size());
    for(auto i = 0u; i < population.size(); ++i)
    {
        ranked.emplace_back(population[i], population[i].fitness());
    }

    // Reverse sort: greatest relative fitness to lowest
    std::sort(std::begin(ranked), std::end(ranked),
        [](RankedDNA const & lhs, RankedDNA const & rhs)
        {
            return lhs.second > rhs.second;
        }
    );

    // ...
#endif
}


}
