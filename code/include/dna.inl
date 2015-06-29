#ifndef DNA_INL
#define DNA_INL

#include "dna.hpp"

template <typename T, typename DNAType>
DNA<T, DNAType>::DNA(Subject subject):
    m_subject(subject), m_fitness(0)
{

}

template <typename T, typename DNAType>
typename DNA<T, DNAType>::Subject DNA<T, DNAType>::getSubject()
{
    return m_subject;
}

template <typename T, typename DNAType>
typename DNA<T, DNAType>::Subject DNA<T, DNAType>::getSubject() const
{
    return m_subject;
}

template <typename T, typename DNAType>
void DNA<T, DNAType>::setSubject(Subject subject)
{
    m_subject = subject;
}

template <typename T, typename DNAType>
typename DNA<T, DNAType>::Fitness DNA<T, DNAType>::getFitness() const
{
    return m_fitness;
}

#endif //DNA_INL
