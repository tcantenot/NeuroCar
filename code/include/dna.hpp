#ifndef NEURO_CAR_DNA_HPP
#define NEURO_CAR_DNA_HPP

#include <memory>

namespace NeuroCar {

template <typename T>
using Individual = std::shared_ptr<T>;

template <typename T, typename ...Args>
Individual<T> createIndividual(Args && ...args)
{
    return std::make_shared<T>(std::forward<Args>(args)...);
}

template <typename T, typename DNAType>
class DNA
{
    public:
        using Fitness = double;
        using MutationRate = double;
        using Subject = Individual<T>;

    public:
        DNA(Subject subject);

        Subject getSubject();
        Subject getSubject() const;
        void setSubject(Subject subject);
        Fitness getFitness() const;

        virtual void randomize(std::size_t seed) = 0;
        virtual Fitness computeFitness(std::size_t ngen = 0) = 0;
        virtual void reset() = 0;
        virtual Subject crossover(DNAType const & partner) const = 0;
        virtual void mutate(MutationRate mutationRate) = 0;

    protected:
        Subject m_subject;
        Fitness m_fitness;
};

}

#include "dna.inl"

#endif //NEURO_CAR_DNA_HPP
