#include "evolution.hpp"

#include <omp.h>

#include <algorithm>
#include <cassert>
#include <chrono>

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
    EvolutionParams params
)
{
    using Fitness = typename DNAType::Fitness;
    using MutationRate = typename DNAType::MutationRate;

    if(params.procInfo.rank == 0)
    {
        assert(dnas.size() > 0);
        assert(nextGen.size() == dnas.size());

        // Compute the fitness of the dnas
        std::size_t const popSize = dnas.size();
        #pragma omp parallel
        {
            #pragma omp for schedule(dynamic, 1)
            for(auto i = 0u; i < popSize; ++i)
            {
                dnas[i].computeFitness();
            }
        }
    }

    // TODO: do this in parallel
    // -> map reduce
    Fitness cumulativeFitness = 0.0;

    if(params.procInfo.rank == 0)
    {
        for(auto const & dna: dnas)
        {
            cumulativeFitness += dna.getFitness();
        }

        std::cout << "Seq: " << cumulativeFitness << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    {
        int32_t id = params.procInfo.rank;
        int32_t p  = params.procInfo.nproc;
        std::size_t n = dnas.size();
        std::size_t size = BLOCK_SIZE(id, p, n);
        std::size_t offset = BLOCK_LOW(id, p, n);

        Fitness localAccu = 0.0;
        for(auto i = offset; i < size; ++i)
        {
            localAccu += dnas[i].getFitness();
        }

        MPI_Reduce(&localAccu, &cumulativeFitness, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    if(params.procInfo.rank == 0)
    {
        std::cout << "MPI: " << cumulativeFitness << std::endl;
    }

    // TODO: stop on fitness threshold?
    //Fitness avgFitness = cumulativeFitness / static_cast<Fitness>(dnas.size());
    //std::cout << "Average fitness = " << avgFitness << std::endl;

    if(params.procInfo.rank == 0)
    {

        // Create the mating pool
        #pragma omp parallel
        {
            #pragma omp for schedule(dynamic, 1)
            for(auto i = 0u; i < dnas.size(); ++i)
            {
                auto const & dna = dnas[i];
                matingPool[i] = RankedDNA<DNAType>(
                    std::cref(dna), dna.getFitness() / cumulativeFitness
                );
            }
        }

        // Reverse sort: greatest relative fitness to lowest
        std::sort(std::begin(matingPool), std::end(matingPool),
            [](RankedDNA<DNAType> const & lhs, RankedDNA<DNAType> const & rhs)
            {
                return lhs.score > rhs.score;
            }
        );

        // TODO: do this in parallel
        // -> map reduce
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
            std::mt19937_64 rng((omp_get_thread_num() + 1) * static_cast<uint64_t>(
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

            // Create next generation
            #pragma omp for schedule(dynamic, 1)
            for(auto i = 0u; i < dnas.size(); ++i)
            {
                DNAType const & parentA = selectParent(matingPool);
                DNAType const & parentB = selectParent(matingPool);

                //TODO:
                // - When to remove the T entity?
                DNAType childDNA(parentA.crossover(parentB));
                childDNA.mutate(mutationRate);

                nextGen[i] = std::move(childDNA);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

}

template <typename DNAType, typename T>
DNAs<DNAType> evolve(
    Population<T> const & population,
    std::size_t ngenerations,
    EvolutionParams params
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
    MatingPool<DNAType> matingPool;
    DNAs<DNAType> nextGen;

    if(params.procInfo.rank == 0)
    {
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
        matingPool.reserve(population.size());
        for(auto const & dna: dnas)
        {
            matingPool.emplace_back(std::cref(dna), 0.0);
        }

        // Initialize the container for the next generation
        nextGen.resize(population.size());
    }

    // Evolve
    for(auto i = 0u; i < ngenerations; ++i)
    {
        std::cout << "Generation " << i << std::endl;

        evolution(dnas, nextGen, matingPool, params);
        if(params.procInfo.rank == 0)
        {
            std::swap(nextGen, dnas);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if(params.procInfo.rank == 0)
    {
        // Evaluate the last generation
        std::size_t const popSize = nextGen.size();
        #pragma omp parallel
        {
            #pragma omp for schedule(dynamic, 1)
            for(auto i = 0u; i < popSize; ++i)
            {
                nextGen[i].computeFitness();
            }
        }

        std::sort(std::begin(nextGen), std::end(nextGen),
            [](DNAType const & lhs, DNAType const & rhs)
            {
                return lhs.getFitness() < rhs.getFitness();
            }
        );
    }

    return nextGen;
}

}
