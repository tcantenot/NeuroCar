#ifndef NEURO_CAR_EVOLUTION_INL
#define NEURO_CAR_EVOLUTION_INL

#include "evolution.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cassert>
#include <chrono>
#include <functional>
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
void evolution(
    DNAs<DNAType> & dnas,
    DNAs<DNAType> & nextGen,
    MatingPool<DNAType> & matingPool,
    EvolutionParams<DNAType> const & params
)
{
    assert(dnas.size() > 0);
    assert(nextGen.size() == dnas.size());
    assert(params.elitism <= dnas.size());

    using Fitness = typename DNAType::Fitness;
    using MutationRate = typename DNAType::MutationRate;

    std::size_t const popSize = dnas.size();

#if 0
    std::cout << "BEFORE" << std::endl;
    for(auto n = 0u; n < dnas.size(); ++n)
        std::cout << "Individual " << n << ": " << ((dnas[n].getSubject())) << " (fitness: " << dnas[n].getFitness() << ")" << std::endl;

    for(auto i = 0u; i < 5; ++i)
    {
        std::cout << "FITNESS: " << dnas[0].computeFitness() << std::endl;
        dnas[0].reset();
    }
#endif

    // Compute the fitness of the dnas
    #pragma omp parallel for schedule(dynamic, 1)
    for(auto i = 0u; i < popSize; ++i)
    {
        dnas[i].computeFitness();
    }

#if 0
    std::cout << "AFTER" << std::endl;
    for(auto n = 0u; n < dnas.size(); ++n)
        std::cout << "Individual " << n << ": " << ((dnas[n].getSubject())) << " (fitness: " << dnas[n].getFitness() << ")" << std::endl;
#endif

    Fitness cumulativeFitness = 0.0;
    #pragma omp parallel for reduction(+:cumulativeFitness)
    for(auto i = 0u; i < popSize; ++i)
    {
        cumulativeFitness += dnas[i].getFitness();
    }

    // Create the mating pool
    #pragma omp parallel for schedule(dynamic, 1)
    for(auto i = 0u; i < dnas.size(); ++i)
    {
        auto const & dna = dnas[i];
        matingPool[i] = RankedDNA<DNAType>(
            std::cref(dna), dna.getFitness() / cumulativeFitness
        );
    }

    // Reverse sort: greatest relative fitness to lowest
    std::sort(std::begin(matingPool), std::end(matingPool),
        [](RankedDNA<DNAType> const & lhs, RankedDNA<DNAType> const & rhs)
        {
            return lhs.score > rhs.score;
        }
    );


    // Compute score of dnas
    Fitness cumulativeScore = 0.0;
    for(auto & dna: matingPool)
    {
        cumulativeScore += dna.score;
        dna.score = cumulativeScore;
    }

#if 0
    {
        //auto n = matingPool.size() - 1;
        //std::cout << "Best Individual: " << (&matingPool[0].dna) << " (fitness: " << matingPool[0].dna.get().getFitness() << ", relative fitness:" << matingPool[0].score << ")" << std::endl;
        //std::cout << "Worst Individual: " << (&matingPool[n].dna) << " (fitness: " << matingPool[n].dna.get().getFitness() << ", relative fitness:" << matingPool[n].score << ")" << std::endl;


        for(auto n = 0u; n < matingPool.size(); ++n)
            std::cout << "Individual " << n << ": " << ((matingPool[n].dna.get().getSubject())) << " (fitness: " << matingPool[n].dna.get().getFitness() << ", relative fitness:" << matingPool[n].score << ")" << std::endl;
    }
#endif

    // Reproduce
    MutationRate const mutationRate = params.mutationRate;
    #pragma omp parallel
    {
        // Create a random generator per thread

        #ifdef _OPENMP
        std::size_t threadNum = omp_get_thread_num();
        #else
        std::size_t threadNum = 0;
        #endif

        std::mt19937_64 rng((threadNum + 1) * static_cast<uint64_t>(
            std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())
        ));
        std::uniform_real_distribution<double> random(0.0, 1.0);

        // Parent selection lambda
        auto const selectParent =
        [&rng, &random](MatingPool<DNAType> & matingPool) -> DNAType const &
        {
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


        // Elitism: keep the best individuals of the previous generation
        #pragma omp for schedule(dynamic, 1)
        for(auto i = 0u; i < params.elitism; ++i)
        {
            nextGen[i] = matingPool[i].dna;
            nextGen[i].reset();
        }

        // Create next generation
        #pragma omp for schedule(dynamic, 1)
        for(auto i = params.elitism; i < dnas.size(); ++i)
        {
            DNAType const & parentA = selectParent(matingPool);
            DNAType const & parentB = selectParent(matingPool);

            DNAType childDNA(parentA.crossover(parentB));
            childDNA.mutate(mutationRate);

            nextGen[i] = std::move(childDNA);
        }
    }
}

}

template <typename DNAType, typename T>
DNAs<DNAType> evolve(
    Population<T> const & population,
    std::size_t ngenerations,
    EvolutionParams<DNAType> const & params
)
{
    static_assert(
        std::is_base_of<DNA<T, DNAType>, DNAType>::value,
        "DNAType must inherit from DNA<T, DNAType>"
    );

    static_assert(
        std::is_default_constructible<DNAType>::value,
        "DNAType must be default constructible"
    );

    DNAs<DNAType> dnas;
    dnas.reserve(population.size());

    int seed = time(NULL);

    // Create initial DNAs
    for(auto i = 0u; i < population.size(); ++i)
    {
        assert(population[i] != nullptr);
        DNAType dna(population[i]);
        dna.randomize(seed+i);
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

        evolution(dnas, nextGen, matingPool, params);

        params.postGenHook(i, dnas);

        std::swap(nextGen, dnas);

#if 1
        typename DNA<T, DNAType>::Fitness maxFitness = 0.0;
        for(auto const & dna: dnas)
        {
            auto f = dna.getFitness();
            maxFitness = std::max(f, maxFitness);
        }
        std::cout << "Max fitness: " << maxFitness << std::endl;
#endif
    }

    // Evaluate the last generation
    std::swap(nextGen, dnas);
    std::size_t const popSize = nextGen.size();
    #pragma omp parallel for schedule(dynamic, 1)
    for(auto i = 0u; i < popSize; ++i)
    {
        nextGen[i].computeFitness();
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

#endif //NEURO_CAR_EVOLUTION_INL
