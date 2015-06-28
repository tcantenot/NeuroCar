#include <self_driving_car.hpp>
#include <renderer.hpp>

#include <random>

namespace NeuroCar {

SelfDrivingCar::SelfDrivingCar():
    m_car(nullptr),
    m_neuroController()
{

}

NeuroController & SelfDrivingCar::getNeuroController()
{
    return m_neuroController;
}

NeuroController const & SelfDrivingCar::getNeuroController() const
{
    return m_neuroController;
}

SelfDrivingCar::NeuralNetwork & SelfDrivingCar::getNeuralNetwork()
{
    return m_neuroController.getNeuralNetwork();
}

void SelfDrivingCar::setNeuroController(NeuroController nc)
{
    m_neuroController = nc;
}

SelfDrivingCar::NeuralNetwork const & SelfDrivingCar::getNeuralNetwork() const
{
    return m_neuroController.getNeuralNetwork();
}


b2Vec2 const & SelfDrivingCar::getDestination() const
{
    return m_destination;
}

void SelfDrivingCar::setDestination(b2Vec2 destination)
{
    m_destination = destination;
    m_neuroController.setDestination(destination);
}

void SelfDrivingCar::setWorldSeed(uint32_t seed)
{
    m_worldSeed = seed;
}

uint32_t SelfDrivingCar::getWorldSeed()
{
    return m_worldSeed;
}

std::shared_ptr<Car> const & SelfDrivingCar::getCar() const
{
    return m_car;
}

void SelfDrivingCar::setCar(std::shared_ptr<Car> car)
{
    m_car = car;
    m_car->setController(&m_neuroController);
}

SelfDrivingCarDNA::SelfDrivingCarDNA(Subject subject):
    DNA(subject),
    m_fitness(0.0)
{

}

void SelfDrivingCarDNA::randomize(std::size_t seed)
{
    auto car = this->getSubject();
    NeuroController & nc = car->getNeuroController();
    SelfDrivingCar::NeuralNetwork & nn = nc.getNeuralNetwork();
    nn.setSeed(seed);
    nn.synthetize();
}

SelfDrivingCarDNA::Fitness SelfDrivingCarDNA::computeFitness()
{
    uint32_t const worldWidth = 100;
    uint32_t const worldHeight = 80;
    uint32_t const nbObstacles = 15;

    // Create world lambda
    #if CAR_PHYSICS_GRAPHIC_MODE_SFML
    Renderer r(8, worldWidth, worldHeight);
    auto const createWorld = [&r](
        uint32_t w, uint32_t h, uint32_t nbObstacles, uint32_t seed
    )
    {
        World * world = new World(8, 3, &r, 10, 2);
        world->addBorders(w, h);
        //world->randomize(w, h, nbObstacles, seed);
        return world;
    };
    #else
    auto const createWorld = [](
        uint32_t w, uint32_t h, uint32_t nbObstacles, uint32_t seed
    )
    {
        World * world = new World(8, 3, 10);
        world->addBorders(w, h);
        //world->randomize(w, h, nbObstacles, seed);
        return world;
    };
    #endif

    uint32_t seed = this->getSubject()->getWorldSeed();

    World * world = createWorld(worldWidth, worldHeight, nbObstacles, seed);

    std::shared_ptr<Car> car = this->getSubject()->getCar();

    // Generate worlds until one is valid
#if 0
    while(world->willCollide(car))
    {
        delete world;
        world = createWorld(worldWidth, worldHeight, nbObstacles, ++seed);
    }
#endif

    world->addRequiredDrawable(car);

    world->run();

    b2Vec2 pos = car->getPos();

    b2Vec2 destination = this->getSubject()->getDestination();

    static auto const distance = [](b2Vec2 const & lhs, b2Vec2 const & rhs)
    {
        return std::sqrt(std::pow(rhs.x - lhs.x, 2) + std::pow(rhs.y - lhs.y, 2));
    };

    float const maxDistance = distance(b2Vec2(0.0, 0.0), b2Vec2(worldWidth, worldHeight));

    Fitness fitness = 1.0 - (distance(pos, destination) / maxDistance);
    m_fitness = fitness;

    delete world;

    return fitness;
}

SelfDrivingCarDNA::Fitness SelfDrivingCarDNA::getFitness() const
{
    return m_fitness;
}

void SelfDrivingCarDNA::reset()
{
    // Copy and reset previous car
    m_subject->setCar(m_subject->getCar()->cloneInitial());
}

SelfDrivingCarDNA::Subject SelfDrivingCarDNA::crossover(SelfDrivingCarDNA const & partner) const
{
    using NeuralNetwork = SelfDrivingCar::NeuralNetwork;

    assert(m_subject);

    Subject child = createIndividual<NeuroCar::SelfDrivingCar>(*m_subject);

    // Copy and reset previous car
    child->setCar(m_subject->getCar()->cloneInitial());

    static std::random_device rd;
    static std::default_random_engine rng(rd());
    std::uniform_real_distribution<double> random(0, 1.0);

    NeuralNetwork const & parentAGenes = m_subject->getNeuralNetwork();
    NeuralNetwork const & parentBGenes = partner.getSubject()->getNeuralNetwork();

    auto const selectParent = [&parentAGenes, &parentBGenes](float r)
        -> NeuralNetwork const &
    {
        return r < 0.5 ? parentAGenes : parentBGenes;
    };

    auto & childGenes = child->getNeuralNetwork();

    // Generate new ADN by combining the parents' DNAs
    auto const & nnshape = parentAGenes.getShape();
    for(auto l = 0u; l < nnshape.size()-1; ++l)
    {
        auto I = nnshape[l];
        auto J = nnshape[l+1];

        for(auto j = 0u; j < J; ++j)
        {
            for(auto i = 0u; i < I; ++i)
            {
                NeuralNetwork const & nn = selectParent(random(rng));
                childGenes.setWeight(l, i, j, nn.getWeight(l, i, j));
            }

            NeuralNetwork const & nn = selectParent(random(rng));
            // j because bias vectors start at layer 1
            childGenes.setBias(l, j, nn.getBias(l, j));
        }
    }

    return child;
}

void SelfDrivingCarDNA::mutate(MutationRate mutationRate)
{
    static std::random_device rd;
    static std::default_random_engine rng(rd());
    std::uniform_real_distribution<double> lottery(0.0, 1.0);
    std::uniform_real_distribution<MutationRate> mutation(-1.0, 1.0);

    auto car = this->getSubject();
    assert(car);

    auto & nn = car->getNeuralNetwork();

    auto const & nnshape = nn.getShape();
    for(auto l = 0u; l < nnshape.size()-1; ++l)
    {
        auto I = nnshape[l];
        auto J = nnshape[l+1];

        for(auto j = 0u; j < J; ++j)
        {
            for(auto i = 0u; i < I; ++i)
            {
                MutationRate r = lottery(rng);
                if(r < mutationRate)
                {
                    auto w = nn.getWeight(l, i, j) + mutation(rng);
                    nn.setWeight(l, i, j, w);
                }
            }

            MutationRate r = lottery(rng);
            if(r < mutationRate)
            {
                // j because bias vectors start at layer 1
                auto b = nn.getBias(l, j) + mutation(rng);
                nn.setBias(l, j, b);
            }
        }
    }
}

}
