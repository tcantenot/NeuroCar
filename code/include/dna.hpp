#ifndef NEURO_CAR_DNA_HPP
#define NEURO_CAR_DNA_HPP

#include <memory>

namespace NeuroCar {

template <typename T, typename DNAType>
class DNA
{
    public:
        using Fitness = double;
        using MutationRate = double;
        using Subject = T;

    public:

        DNA(T * subject): m_subject(subject) { }

        DNA(std::unique_ptr<T> && subject): m_subject(std::move(subject)) { }

        template <typename ...Args>
        DNA(Args && ...args): m_subject(new T{std::forward<Args>(args)...}) { }

        T * getSubject() { return m_subject.get(); }
        T const * getSubject() const { return m_subject.get(); }
        void setSubject(T * subject) { m_subject.reset(subject); }
        void setSubject(std::unique_ptr<T> && subject) { m_subject = std::move(subject); }

        virtual void randomize() = 0;
        virtual Fitness computeFitness() = 0;
        virtual Fitness getFitness() const = 0;
        virtual std::unique_ptr<T> crossover(DNAType const & partner) const = 0;
        virtual void mutate(MutationRate mutationRate) = 0;

    protected:
        std::unique_ptr<T> m_subject;
};

}

#endif //NEURO_CAR_DNA_HPP
