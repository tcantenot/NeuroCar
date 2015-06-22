#ifndef NEURO_CAR_DNA_INL
#define NEURO_CAR_DNA_INL

#include "dna.hpp"

namespace NeuroCar {

template <typename T, typename DNAType>
DNA<T, DNAType>::DNA(Subject subject): m_subject(std::move(subject))
{

}

template <typename T, typename DNAType>
T * DNA<T, DNAType>::getSubject()
{
    return m_subject.get();
}

template <typename T, typename DNAType>
T const * DNA<T, DNAType>::getSubject() const
{
    return m_subject.get();
}

template <typename T, typename DNAType>
void DNA<T, DNAType>::setSubject(Subject subject)
{
    m_subject = std::move(subject);
}

}

#endif //NEURO_CAR_DNA_INL
