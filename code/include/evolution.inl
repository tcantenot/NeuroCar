#include "evolution.hpp"

#include <algorithm>
#include <cassert>

#include <iostream>

namespace NeuroCar {

namespace {

template <typename DNAType>
DNAs<DNAType> evolution(DNAs<DNAType> & dnas)
{
    #if 0
    static_assert(
        std::is_base_of<DNA<DNAType::Subject, DNAType>>::value,
        "DNAType must inherits from DNA"
    );
    #endif

    assert(dnas.size() > 0);

    // Train
    // TODO: do this in parallel
    for(auto & dna: dnas)
    {
        dna.computeFitness();
    }

    // Compute the fitness of the dnas

    using Fitness = typename DNAType::Fitness;

    // TODO: do this in parallel
    // -> map reduce
    Fitness cumulativeFitness = 0.0;
    for(auto const & dna: dnas)
    {
        cumulativeFitness += dna.getFitness();
    }

    Fitness avgFitness = cumulativeFitness / static_cast<Fitness>(dnas.size());

    // TODO: stop on fitness threshold?
    //std::cout << "Average fitness = " << avgFitness << std::endl;


    // Create the mating pool

    //TODO: create a struct for RankedDNA -> better readability
    // + use std::ref<DNA> ?
    using RankedDNA  = std::pair<std::reference_wrapper<const DNAType>, Fitness>;
    using MatingPool = std::vector<RankedDNA>;

    MatingPool matingPool;
    matingPool.reserve(dnas.size());

    for(auto const & dna: dnas)
    {
        matingPool.emplace_back(std::cref(dna), dna.getFitness() / cumulativeFitness);
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
    //
    static auto const selectParent = [&matingPool]() -> DNAType const &
    {
        static std::random_device rd;
        static std::mt19937 rng(rd());
        static std::uniform_real_distribution<double> random(0.0, 1.0);

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

    DNAs<DNAType> nextGeneration;
    nextGeneration.reserve(dnas.size());

    for(auto i = 0u; i < dnas.size(); ++i)
    {
        DNAType const & parentA = selectParent();
        DNAType const & parentB = selectParent();

        //TODO:
        // - When to remove the T entity?
        auto child = parentA.crossover(parentB);
        DNAType childDNA(child);
        childDNA.mutate(mutationRate);

        nextGeneration.emplace_back(std::move(childDNA));
    }

    return nextGeneration;
}

}

template <typename DNAType, typename T>
DNAs<DNAType> evolve(Population<T> const & population, std::size_t ngenerations)
{
    DNAs<DNAType> dnas;
    dnas.reserve(population.size());

    // Create initial DNAs
    for(auto i = 0u; i < population.size(); ++i)
    {
        assert(population[i] != nullptr);
        DNAType dna(population[i]);
        dna.randomize();
        dnas.emplace_back(std::move(dna));
    }

    assert(dnas.size() == population.size());

    // Evolve
    for(auto i = 0u; i < ngenerations; ++i)
    {
        std::cout << "Generation " << i << std::endl;
        dnas = evolution(dnas);
    }

    for(auto & dna: dnas)
    {
        dna.computeFitness();
    }

    // Reverse sort: greatest fitness to lowest
    std::sort(std::begin(dnas), std::end(dnas),
        [](DNAType const & lhs, DNAType const & rhs)
        {
            return lhs.getFitness() < rhs.getFitness();
        }
    );

    return dnas;
}

}
