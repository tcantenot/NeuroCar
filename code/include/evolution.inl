#include "evolution.hpp"

#include <algorithm>
#include <cassert>

#include <iostream>

namespace NeuroCar {

namespace {

template <typename DNAType>
struct RankedDNA
{
    using DNARef = std::reference_wrapper<const DNAType>;
    using Score  = typename DNAType::Fitness;

    DNARef dna;
    Score score;

    RankedDNA(DNARef d, Score s): dna(d), score(s) { }
};

template <typename DNAType>
using MatingPool = std::vector<RankedDNA<DNAType>>;


template <typename DNAType>
void evolution(DNAs<DNAType> & dnas, DNAs<DNAType> & nextGen, MatingPool<DNAType> & matingPool)
{
    static_assert(
        std::is_default_constructible<DNAType>::value,
        "DNAType must be default constructible"
    );

    static_assert(
        std::is_base_of<DNA<typename DNAType::Subject, DNAType>, DNAType>::value,
        "DNAType must inherits from DNA<T, DNAType>"
    );

    assert(dnas.size() > 0);
    assert(nextGen.size() == dnas.size());

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
    for(auto i = 0u; i < dnas.size(); ++i)
    {
        auto const & dna = dnas[i];
        matingPool[i] = RankedDNA<DNAType>(std::cref(dna), dna.getFitness() / cumulativeFitness);
    }

    // Reverse sort: greatest relative fitness to lowest
    std::sort(std::begin(matingPool), std::end(matingPool),
        [](RankedDNA<DNAType> const & lhs, RankedDNA<DNAType> const & rhs)
        {
            return lhs.score > rhs.score;
        }
    );

    Fitness cumulativeScore = 0.0;
    for(auto & dna: matingPool)
    {
        cumulativeScore += dna.score;
        dna.score = cumulativeScore;
    }

    // Reproduce

    static auto const selectParent = [](MatingPool<DNAType> & matingPool) -> DNAType const &
    {
        static std::random_device rd;
        static std::mt19937 rng(rd());
        static std::uniform_real_distribution<double> random(0.0, 1.0);

        Fitness r = random(rng);
        for(auto i = 0u; i < matingPool.size()-1; ++i)
        {
            if(matingPool[i].score > r)
            {
                return matingPool[i].dna;
            }
        }

        return matingPool[matingPool.size()-1].dna;
    };

    using MutationRate = typename DNAType::MutationRate;

    MutationRate const mutationRate = 0.01;

    for(auto i = 0u; i < dnas.size(); ++i)
    {
        DNAType const & parentA = selectParent(matingPool);
        DNAType const & parentB = selectParent(matingPool);

        //TODO:
        // - When to remove the T entity?
        auto child = parentA.crossover(parentB);
        DNAType childDNA(child);
        childDNA.mutate(mutationRate);

        nextGen[i] = std::move(childDNA);
    }
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

    // Initialize a dummy mating pool
    MatingPool<DNAType> matingPool;
    matingPool.reserve(population.size());
    for(auto const & dna: dnas)
    {
        matingPool.emplace_back(std::cref(dna), 0.0);
    }

    // Initialize the container for the next generation
    DNAs<DNAType> nextGen;
    nextGen.resize(population.size());

    // Evolve
    for(auto i = 0u; i < ngenerations; ++i)
    {
        std::cout << "Generation " << i << std::endl;

        evolution(dnas, nextGen, matingPool);
        std::swap(nextGen, dnas);
    }

    // Evaluate the last generation
    for(auto & dna: nextGen)
    {
        dna.computeFitness();
    }

    std::sort(std::begin(nextGen), std::end(nextGen),
        [](DNAType const & lhs, DNAType const & rhs)
        {
            return lhs.getFitness() < rhs.getFitness();
        }
    );

    return nextGen;
}

}
