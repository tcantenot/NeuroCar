#ifndef NEURO_CAR_DNA_INL
#define NEURO_CAR_DNA_INL

#include "dna.hpp"

namespace NeuroCar {

template <typename T, typename DNAType>
DNA<T, DNAType>::DNA(Subject subject): m_subject(subject)
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

}

#endif //NEURO_CAR_DNA_INL
