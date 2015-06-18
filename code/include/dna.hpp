#ifndef NEURO_CAR_DNA_HPP
#define NEURO_CAR_DNA_HPP

namespace NeuroCar {

template <typename T, typename DNAType>
class DNA
{
    public:
        using Fitness = double;
        using MutationRate = double;

    public:

        DNA(T * subject): m_subject(subject) { }

        T * getSubject() { return m_subject; }
        T const * getSubject() const { return m_subject; }

        virtual void randomize() = 0;
        virtual Fitness fitness() const = 0;
        virtual T * crossover(DNAType const & partner) const = 0;
        virtual void mutate(MutationRate mutationRate) = 0;

    protected:
        T * m_subject;
};

}

#endif //NEURO_CAR_DNA_HPP
