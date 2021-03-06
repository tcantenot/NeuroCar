#ifndef EVOLUTION_INL
#define EVOLUTION_INL

#include "evolution.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cassert>
#include <chrono>
#include <functional>

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
    std::size_t ngen,
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

    // Compute the fitness of the dnas
    #pragma omp parallel for schedule(dynamic, 1)
    for(auto i = 0u; i < popSize; ++i)
    {
        dnas[i].computeFitness(ngen);
    }

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

            DNAType childDNA;
            childDNA.setSubject(parentA.crossover(parentB));
            childDNA.init(params.dnaParams);
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

    int seed = 42;//time(NULL);

    // Create initial DNAs
    for(auto i = 0u; i < population.size(); ++i)
    {
        assert(population[i] != nullptr);
        DNAType dna;
        dna.setSubject(population[i]);
        dna.init(params.dnaParams);
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
        params.preGenHook(i, dnas);

        evolution(i, dnas, nextGen, matingPool, params);

        params.postGenHook(i, dnas);

        std::swap(nextGen, dnas);
    }

    // Evaluate the last generation
    std::swap(nextGen, dnas);
    std::size_t const popSize = nextGen.size();

    params.preGenHook(ngenerations, nextGen);

    #pragma omp parallel for schedule(dynamic, 1)
    for(auto i = 0u; i < popSize; ++i)
    {
        nextGen[i].computeFitness(ngenerations);
    }

    params.postGenHook(ngenerations, nextGen);

    std::sort(std::begin(nextGen), std::end(nextGen),
        [](DNAType const & lhs, DNAType const & rhs)
        {
            return lhs.getFitness() < rhs.getFitness();
        }
    );

    return nextGen;
}

#endif //EVOLUTION_INL
