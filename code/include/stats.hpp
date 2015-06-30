#ifndef STATS_HPP
#define STATS_HPP

#include <algorithm>
#include <fstream>

class StatsAll
{
    public:
        using Fitness = double;

        StatsAll(std::string const & filename):
            m_file(filename, std::ios::out | std::ios::trunc),
            m_cumulativeFitness(0.0)
        {

        }

        template <typename DNAType>
        void operator()(std::size_t i, DNAs<DNAType> const & dnas)
        {
            auto const & bestDNA = *std::max_element(dnas.begin(), dnas.end(),
                [](DNAType const & lhs, DNAType const & rhs)
                {
                    return lhs.getFitness() < rhs.getFitness();
                }
            );

            Fitness maxFitness = bestDNA.getFitness();
            m_cumulativeFitness += maxFitness;
            Fitness mean = m_cumulativeFitness / static_cast<Fitness>(i+1);
            m_file << maxFitness << ", " << mean << std::endl;
        }

    private:
        std::fstream m_file;
        Fitness m_cumulativeFitness;
};

class StatsLastN
{
    public:
        using Fitness = double;

        StatsLastN(std::string const & filename, std::size_t n):
            m_file(filename, std::ios::out | std::ios::trunc),
            m_history(n, 0.0),
            m_index(0)
        {

        }

        template <typename DNAType>
        void operator()(std::size_t i, DNAs<DNAType> const & dnas)
        {
            auto const & bestDNA = *std::max_element(dnas.begin(), dnas.end(),
                [](DNAType const & lhs, DNAType const & rhs)
                {
                    return lhs.getFitness() < rhs.getFitness();
                }
            );

            Fitness maxFitness = bestDNA.getFitness();

            m_history[m_index] = maxFitness;
            m_index = (m_index + 1) % m_history.size();

            Fitness cumulativeFitness = std::accumulate(
                m_history.begin(), m_history.end(), Fitness(0.0)
            );

            std::size_t n = std::min(i+1, m_history.size());

            Fitness mean = cumulativeFitness / static_cast<Fitness>(n);

            m_file << maxFitness << ", " << mean << std::endl;
        }

    private:
        std::fstream m_file;
        std::vector<Fitness> m_history;
        std::size_t m_index;
};



#endif //STATS_HPP
