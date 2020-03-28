
#include "DatasetCreator.h"

namespace DatasetCreator
{
Dataset GenerateXDataset(unsigned size)
{
    Dataset      dataset({ std::vector<std::vector<double>>(size), std::vector<std::vector<double>>(size) });
    const double width = SCALE;

    for (unsigned i = 0; i < size; i++)
    {
        double x      = (rand() * width * 2.0) / RAND_MAX - width;
        double y      = (rand() * width * 2.0) / RAND_MAX - width;
        double noiseX = x + NOISE * (rand() * 2.0 / RAND_MAX - 1.0);
        double noiseY = y + NOISE * (rand() * 2.0 / RAND_MAX - 1.0);

        dataset.first[i]  = { x, y };
        dataset.second[i] = { x };
    }

    return dataset;
}

Dataset GenerateQuadrantsDataset(unsigned size)
{
    Dataset      dataset({ std::vector<std::vector<double>>(size), std::vector<std::vector<double>>(size) });
    const double width = SCALE;

    for (unsigned i = 0; i < size; i++)
    {
        double x      = (rand() * width * 2.0) / RAND_MAX - width;
        double y      = (rand() * width * 2.0) / RAND_MAX - width;
        double noiseX = x + NOISE * (rand() * 2.0 / RAND_MAX - 1.0);
        double noiseY = y + NOISE * (rand() * 2.0 / RAND_MAX - 1.0);
        double value  = noiseX * noiseY > 0.0 ? 1.0 : -1.0;

        dataset.first[i]  = { x, y };
        dataset.second[i] = { value };
    }

    return dataset;
}

Dataset GenerateRingDataset(unsigned size)
{
    Dataset      dataset({ std::vector<std::vector<double>>(size), std::vector<std::vector<double>>(size) });
    const double width = SCALE;

    for (unsigned i = 0; i < size; i++)
    {
        double x      = (rand() * width * 2.0) / RAND_MAX - width;
        double y      = (rand() * width * 2.0) / RAND_MAX - width;
        double noiseX = x + NOISE * (rand() * 2.0 / RAND_MAX - 1.0);
        double noiseY = y + NOISE * (rand() * 2.0 / RAND_MAX - 1.0);
        double value  = std::sqrt((noiseX * noiseX) + (noiseY * noiseY));

        value = value > SCALE * 0.5 ? 1.0 : -1.0;

        dataset.first[i]  = { x, y };
        dataset.second[i] = { value };
    }

    return dataset;
}

Dataset GenerateSineDataset(unsigned size)
{
    Dataset      dataset({ std::vector<std::vector<double>>(size), std::vector<std::vector<double>>(size) });
    const double width = SCALE;

    for (unsigned i = 0; i < size; i++)
    {
        double x      = (rand() * width * 2.0) / RAND_MAX - width;
        double y      = (rand() * width * 2.0) / RAND_MAX - width;
        double noiseX = x + NOISE * (rand() * 2.0 / RAND_MAX - 1.0);
        double noiseY = y + NOISE * (rand() * 2.0 / RAND_MAX - 1.0);
        double value  = std::abs(std::sin(noiseX * 3.0) - noiseY) < 0.1 ? 1.0 : -1.0;

        dataset.first[i]  = { x, y };
        dataset.second[i] = { value };
    }

    return dataset;
}

Dataset GenerateCrossDataset(unsigned size)
{
    Dataset      dataset({ std::vector<std::vector<double>>(size), std::vector<std::vector<double>>(size) });
    const double width = SCALE;

    for (unsigned i = 0; i < size; i++)
    {
        double x      = (rand() * width * 2.0) / RAND_MAX - width;
        double y      = (rand() * width * 2.0) / RAND_MAX - width;
        double noiseX = x + NOISE * (rand() * 2.0 / RAND_MAX - 1.0);
        double noiseY = y + NOISE * (rand() * 2.0 / RAND_MAX - 1.0);
        double value  = std::abs(std::abs(noiseX) - std::abs(noiseY)) > 0.1 ? 1.0 : -1.0;

        dataset.first[i]  = { x, y };
        dataset.second[i] = { value };
    }

    return dataset;
}

Dataset GenerateColorQuadrantsDataset(unsigned size)
{
    Dataset      dataset({ std::vector<std::vector<double>>(size), std::vector<std::vector<double>>(size) });
    const double width = SCALE;

    for (unsigned i = 0; i < size; i++)
    {
        double x      = (rand() * width * 2.0) / RAND_MAX - width;
        double y      = (rand() * width * 2.0) / RAND_MAX - width;
        double noiseX = x + NOISE * (rand() * 2.0 / RAND_MAX - 1.0);
        double noiseY = y + NOISE * (rand() * 2.0 / RAND_MAX - 1.0);
        double value  = noiseX * noiseY > 0.0 ? 1.0 : -1.0;

        if (x > 0.0)
        {
            if (y > 0.0)
            {
                dataset.first[i]  = { x, y };
                dataset.second[i] = { 0.0, 0.0, 1.0 };
            }
            else
            {
                dataset.first[i]  = { x, y };
                dataset.second[i] = { 0.0, 1.0, 1.0 };
            }
        }
        else
        {
            if (y > 0.0)
            {
                dataset.first[i]  = { x, y };
                dataset.second[i] = { 1.0, 0.0, 0.0 };
            }
            else
            {
                dataset.first[i]  = { x, y };
                dataset.second[i] = { 0.0, 1.0, 0.0 };
            }
        }
    }

    return dataset;
}

Dataset GenerateColorRingDataset(unsigned size)
{
    Dataset      dataset({ std::vector<std::vector<double>>(size), std::vector<std::vector<double>>(size) });
    const double width = SCALE;

    for (unsigned i = 0; i < size; i++)
    {
        bool inserted = false;
        do
        {
            double x      = (rand() * width * 2.0) / RAND_MAX - width;
            double y      = (rand() * width * 2.0) / RAND_MAX - width;
            double noiseX = x + NOISE * (rand() * 2.0 / RAND_MAX - 1.0);
            double noiseY = y + NOISE * (rand() * 2.0 / RAND_MAX - 1.0);
            double value  = std::sqrt((noiseX * noiseX) + (noiseY * noiseY));

            if (value < 0.3 * width)
            {
                dataset.first[i]  = { x, y };
                dataset.second[i] = { 1.0, 0.0, 0.0 };
                inserted          = true;
            }
            else if (value < 0.6 * width && value > 0.4 * width)
            {
                dataset.first[i]  = { x, y };
                dataset.second[i] = { 0.0, 1.0, 0.0 };
                inserted          = true;
            }
            else if (value < 0.9 * width && value > 0.7 * width)
            {
                dataset.first[i]  = { x, y };
                dataset.second[i] = { 0.0, 0.0, 1.0 };
                inserted          = true;
            }

        } while (!inserted);
    }

    return dataset;
}

Dataset GenerateColorSpiralDataset(unsigned size)
{
    Dataset      dataset({ std::vector<std::vector<double>>(size), std::vector<std::vector<double>>(size) });
    const double width       = SCALE;
    const double SpiralWidth = 0.3;

    for (unsigned i = 0; i < size; i++)
    {
        double x      = (rand() * width * 2.0) / RAND_MAX - width;
        double y      = (rand() * width * 2.0) / RAND_MAX - width;
        double noiseX = x + NOISE * (rand() * 2.0 / RAND_MAX - 1.0);
        double noiseY = y + NOISE * (rand() * 2.0 / RAND_MAX - 1.0);
        double angle  = (std::atan2(noiseY, noiseX) + 3.141516);
        double r      = std::sqrt((noiseX * noiseX) + (noiseY * noiseY));

        double value = r - SpiralWidth * angle / (3.141516 * 2.0);
        while (value > SpiralWidth)
        {
            value -= SpiralWidth;
        }

        if (std::abs(value) < 0.1)
        {
            dataset.first[i]  = { x, y };
            dataset.second[i] = { 1.0, 0.0, 0.0 };
        }
        else
        {
            dataset.first[i]  = { x, y };
            dataset.second[i] = { 0.0, 0.0, 1.0 };
        }
    }

    return dataset;
}

} // namespace DatasetCreator
