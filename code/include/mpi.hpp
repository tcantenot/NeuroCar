#ifndef NEURO_CAR_MPI_HPP
#define NEURO_CAR_MPI_HPP

#define BLOCK_LOW(id, p, n) ((id) * (n) / (p))

#define BLOCK_HIGH(id, p, n) (BLOCK_LOW((id) + 1, p, n) - 1)

#define BLOCK_SIZE(id, p, n) (BLOCK_LOW((id) + 1, p, n) - BLOCK_LOW(id, p, n))

#define BLOCK_OWNER(index, p, n) (((p) * (index) + 1) - 1) / (n))

#if 1

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wcast-qual"

#include <mpi.h>

struct proc_info_t
{
    int32_t rank;
    int32_t nproc;
    char name[MPI_MAX_PROCESSOR_NAME];
    int32_t name_len;
};

class MPIInitializer
{
    public:

        MPIInitializer(): m_info()
        {

        }

        MPIInitializer(int argc, char ** argv): m_info()
        {
            MPIInitializer::init(argc, argv);
        }

        ~MPIInitializer()
        {
            MPIInitializer::finalize();
        }

        proc_info_t getProcInfo() const
        {
            return m_info;
        }

        void init(int argc, char ** argv)
        {
            MPI_Init(&argc, &argv);
            MPI_Comm_size(MPI_COMM_WORLD, &m_info.nproc);
            MPI_Comm_rank(MPI_COMM_WORLD, &m_info.rank);
            MPI_Get_processor_name(m_info.name, &m_info.name_len);
        }

        void finalize()
        {
            MPI_Finalize();
        };

    private:
        proc_info_t m_info;
};

#pragma GCC diagnostic pop

#endif

#endif //NEURO_CAR_MPI_HPP
