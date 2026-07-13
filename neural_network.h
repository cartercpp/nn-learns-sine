//
// Created by cartercpp on 7/12/26.
//

#ifndef NNLEARNSSINE_NEURAL_NETWORK_H
#define NNLEARNSSINE_NEURAL_NETWORK_H

#include <stdexcept>
#include <array>
#include <vector>
#include <utility>
#include <random>

template <int... Is> requires ((sizeof...(Is) > 0) && ((Is > 0) && ...))
class neural_network
{
private:

    using WeightMatrix = std::vector<std::vector<double>>;
    using BiasVector = std::vector<double>;

public:

    // CONSTRUCTORS & STUFF

    explicit neural_network(double learningRate)
        : m_learningRate{learningRate}
    {
        m_weightMatrices.reserve(s_layers);
        m_biasVectors.reserve(s_layers);

        std::random_device rd;
        std::uniform_real_distribution<double> weightDist(-1, 1);

        for (int layer = 1; layer < s_layers; ++layer)
        {
            const int layerSize = s_neuronsPerLayer[layer],
                      prevLayerSize = s_neuronsPerLayer[layer - 1];

            WeightMatrix weights(layerSize, std::vector<double>(prevLayerSize, 0));
            BiasVector biases(layerSize, 0);

            for (int i = 0; i < layerSize; ++i)
                for (int i2 = 0; i2 < prevLayerSize; ++i2)
                    weights[i][i2] = weightDist(rd);

            m_weightMatrices.emplace_back(std::move(weights));
            m_biasVectors.emplace_back(std::move(biases));
        }
    }

    // METHODS & STUFF

    std::vector<double> predict(const std::vector<double>& input) const
    {
        return Forward(input).back();
    }

    void fit(const std::vector<double>& input, const std::vector<double>& target, std::size_t epochs = 1)
    {
        for (std::size_t epoch = 0; epoch < epochs; ++epoch)
            Fit(input, target);
    }

private:

    static constexpr int s_layers = static_cast<int>(sizeof...(Is));
    static constexpr std::array<int, s_layers> s_neuronsPerLayer{Is...};

    static constexpr double TanhDerivative(double x) noexcept
    {
        return 1 - x * x;
    }

    std::vector<std::vector<double>> Forward(const std::vector<double>& input) const
    {
        if (input.size() != s_neuronsPerLayer[0])
            throw std::invalid_argument{"Insufficient # of inputs given"};

        std::vector<std::vector<double>> activations;
        activations.reserve(s_layers);
        activations.push_back(input);

        for (int layer = 0; layer < s_layers - 1; ++layer)
        {
            const std::vector<double>& sourceActivations{activations[layer]};
            const WeightMatrix& sourceWeights{m_weightMatrices[layer]};
            const BiasVector& sourceBiases{m_biasVectors[layer]};
            const int sourceSize = s_neuronsPerLayer[layer];

            const int destSize = s_neuronsPerLayer[layer + 1];
            std::vector<double> destActivations(destSize, 0);

            for (int destNeuron = 0; destNeuron < destSize; ++destNeuron)
            {
                double weightedSum = sourceBiases[destNeuron];

                for (int sourceNeuron = 0; sourceNeuron < sourceSize; ++sourceNeuron)
                    weightedSum += sourceActivations[sourceNeuron] * sourceWeights[destNeuron][sourceNeuron];

                destActivations[destNeuron] = std::tanh(weightedSum);
            }

            activations.emplace_back(std::move(destActivations));
        }

        return activations;
    }

    void Fit(const std::vector<double>& input, const std::vector<double>& target)
    {
        const int activationsInOutput = s_neuronsPerLayer[s_layers - 1];

        if (target.size() != activationsInOutput)
            throw std::invalid_argument{"# of outputs doesn't match"};

        const std::vector<std::vector<double>> activations{Forward(input)};
        const std::vector<double>& prediction{activations.back()};

        std::vector<double> delta(activationsInOutput, 0);
        for (int i = 0; i < activationsInOutput; ++i)
            delta[i] = (prediction[i] - target[i]) * TanhDerivative(prediction[i]);

        for (int layer = s_layers - 2; layer >= 0; --layer)
        {
            const std::vector<double>& sourceActivations{activations[layer]};
            WeightMatrix& sourceWeightsRef{m_weightMatrices[layer]};
            BiasVector& sourceBiasesRef{m_biasVectors[layer]};
            const int sourceSize = s_neuronsPerLayer[layer];
            const int destSize = s_neuronsPerLayer[layer + 1];

            std::vector<double> newDelta(sourceSize, 0);
            for (int sourceNeuron = 0; sourceNeuron < sourceSize; ++sourceNeuron)
            {
                double error = 0;

                for (int destNeuron = 0; destNeuron < destSize; ++destNeuron)
                    error += delta[destNeuron] * sourceWeightsRef[destNeuron][sourceNeuron];

                newDelta[sourceNeuron] = error * ((layer > 0) ? TanhDerivative(sourceActivations[sourceNeuron]) : 1);
            }

            for (int destNeuron = 0; destNeuron < destSize; ++destNeuron)
            {
                for (int sourceNeuron = 0; sourceNeuron < sourceSize; ++sourceNeuron)
                    sourceWeightsRef[destNeuron][sourceNeuron]
                        -= m_learningRate * delta[destNeuron] * sourceActivations[sourceNeuron];

                sourceBiasesRef[destNeuron] -= m_learningRate * delta[destNeuron];
            }

            std::swap(delta, newDelta);
        }
    }

    std::vector<WeightMatrix> m_weightMatrices;
    std::vector<BiasVector> m_biasVectors;
    double m_learningRate;
};

#endif //NNLEARNSSINE_NEURAL_NETWORK_H