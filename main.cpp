#include <bits/stdc++.h>

using namespace std;

int cntInputs = 13;
double learningRate = 0.05 / 151;
double cntLaunches = 40000;
mt19937 rnd(1);

double sigma(double x) {
    return 1 / (1 + exp(-x));
}

double derivativeSigma(double x) {
    return sigma(x) * (1 - sigma(x));
}

double prod(vector<double> &a, vector<double> &b) {
    double ans = 0;
    for (int i = 0; i < a.size(); i++) {
        ans += a[i] * b[i];
    }
    return ans;
}

class Neuron {
public:

    vector<double> weights;

    Neuron() {
        weights = vector<double>(cntInputs + 1);
        for (int i = 0; i < weights.size(); i++) {
            weights[i] = (abs((int) rnd()) % 20000 - 10000) / 10000.0;
        }
    }

    double calculate(vector<double> &inputs) {
        return sigma(prod(weights, inputs));
    }
};

class NeuralNet {
    Neuron outputNeuron;
    vector<Neuron> hiddenNeurons;

public:

    NeuralNet() {
        hiddenNeurons = vector<Neuron>(cntInputs);
        for (int i = 0; i < hiddenNeurons.size(); i++) {
            hiddenNeurons[i] = Neuron();
        }
    }

    double calculate(vector<double> &inputs) {
        vector<double> hiddenAnswers(cntInputs);
        for (int i = 0; i < cntInputs; i++) {
            hiddenAnswers[i] = hiddenNeurons[i].calculate(inputs);
        }
        return outputNeuron.calculate(hiddenAnswers);
    }

    double lossFunc(vector<pair<vector<double>, double>> &data) {
        double ans = 0;
        for (auto[inputs, yRight] : data) {
            double delta = yRight - calculate(inputs);
            ans += delta * delta;
        }
        return ans;
    }

    void recalcGradient(vector<double> &inputs, double rightAnswer, vector<vector<double>> &deltas_w_i_j,
                        vector<double> &deltas_w_out) {
        double predAnswer = calculate(inputs);

        double d_L_d_predAnswer = 2 * (predAnswer - rightAnswer);
        vector<double> h(cntInputs);
        for (int i = 0; i < cntInputs; i++) {
            h[i] = sigma(prod(inputs, hiddenNeurons[i].weights));
        }
        double sumOutput = prod(h, outputNeuron.weights);
        vector<double> d_predAnswer_d_h(cntInputs);
        for (int i = 0; i < cntInputs; i++) {
            d_predAnswer_d_h[i] = outputNeuron.weights[i] * derivativeSigma(sumOutput);
        }
        vector<vector<double>> d_hj_d_wij(inputs.size(), vector<double>(cntInputs));
        for (int j = 0; j < cntInputs; j++) {
            double sumH_j = prod(inputs, hiddenNeurons[j].weights);
            for (int i = 0; i < hiddenNeurons[j].weights.size(); i++) {
                d_hj_d_wij[i][j] = inputs[i] * derivativeSigma(sumH_j);
            }
        }
        for (int j = 0; j < cntInputs; j++) {
            for (int i = 0; i < hiddenNeurons[j].weights.size(); i++) {
                deltas_w_i_j[i][j] -= learningRate * d_L_d_predAnswer * d_predAnswer_d_h[j] * d_hj_d_wij[i][j];
            }
        }
        vector<double> d_predAnswer_d_wout(cntInputs);
        for (int i = 0; i < cntInputs; i++) {
            d_predAnswer_d_wout[i] = h[i] * derivativeSigma(sumOutput);
        }
        for (int i = 0; i < cntInputs; i++) {
            deltas_w_out[i] -= learningRate * d_L_d_predAnswer * d_predAnswer_d_wout[i];
        }
    }

    void addGradient(vector<vector<double>> &deltas_w_i_j, vector<double> &deltas_w_out) {
        for (int j = 0; j < cntInputs; j++) {
            for (int i = 0; i < hiddenNeurons[j].weights.size(); i++) {
                hiddenNeurons[j].weights[i] += deltas_w_i_j[i][j];
            }
        }
        for (int i = 0; i < cntInputs; i++) {
            outputNeuron.weights[i] += deltas_w_out[i];
        }
    }

    void printWeights() {
        cout << "w_i_j = { ";
        for (int j = 0; j < cntInputs; j++) {
            cout << "{ ";
            for (auto w : hiddenNeurons[j].weights) {
                cout << w << ", ";
            }
            if (j + 1 == cntInputs) {
                cout << "} \n";
            } else {
                cout << "}, \n";
            }
        }
        cout << "}";
    }
};

vector<double> split(string splitString, char splitSymb) {
    vector<double> ans;
    string curString;
    for (auto symb : splitString) {
        if (symb == splitSymb) {
            ans.push_back(stod(curString));
            curString = "";
        } else {
            curString += symb;
        }
    }
    if (!curString.empty()) {
        ans.push_back(stod(curString));
    }
    return ans;
}

int main() {
    ifstream inp("D:\\Dmitry\\Kaggle\\Heart Attack clion\\heart.csv");
    string inputString;
    bool isHeaderString = true;
    vector<pair<vector<double>, double>> data; // format: {x: [1, 2, 3], y: 4}
    while (getline(inp, inputString)) {
        if (isHeaderString) {
            isHeaderString = false;
            continue;
        }
        auto splittedInput = split(inputString, ',');
        int y = splittedInput.back();
        splittedInput.back() = 1;
        data.push_back({splittedInput, y});
    }
    shuffle(data.begin(), data.end(), rnd);
    vector<pair<vector<double>, double>> trainSubSet, testSubSet;
    for (int i = 0; i < data.size(); i++) {
        if (i < data.size() / 2) {
            trainSubSet.push_back(data[i]);
        } else {
            testSubSet.push_back(data[i]);
        }
    }
    NeuralNet net;
    double ma = 0;
    for (int i = 0; i < 50; i++) {
        int seed = chrono::high_resolution_clock::now().time_since_epoch().count();
        rnd = mt19937(seed);
        net = NeuralNet();
        for (int j = 0; j < 1000; j++) {
            vector<vector<double>> deltas_w_i_j(cntInputs + 1, vector<double>(cntInputs));
            vector<double> deltas_w_out(cntInputs);
            for (auto[inputs, yRight] : trainSubSet) {
                net.recalcGradient(inputs, yRight, deltas_w_i_j, deltas_w_out);
            }
            net.addGradient(deltas_w_i_j, deltas_w_out);
        }
        int predicted = 0;
        for (auto[inputs, yRight] : testSubSet) {
            double ans = net.calculate(inputs);
            if (ans >= 0.5) {
                predicted += yRight == 1;
            } else {
                predicted += yRight == 0;
            }
        }
        cout << "predicted: " << (double) predicted / testSubSet.size() * 100 << '%' << endl;
        ma = max(ma, (double) predicted / testSubSet.size() * 100);
    }
    cout << "max percent: " << ma << '%';
    exit(0);
    for (int i = 0; i < cntLaunches; i++) {
        vector<vector<double>> deltas_w_i_j(cntInputs + 1, vector<double>(cntInputs));
        vector<double> deltas_w_out(cntInputs);
        for (auto[inputs, yRight] : trainSubSet) {
            net.recalcGradient(inputs, yRight, deltas_w_i_j, deltas_w_out);
        }
        net.addGradient(deltas_w_i_j, deltas_w_out);
        if (i % 100 == 0) {
            cout << "iteration " << i << ": loss function = " << net.lossFunc(trainSubSet) << '\n';
        }
    }
    int predicted = 0;
    for (auto[inputs, yRight] : testSubSet) {
        double ans = net.calculate(inputs);
        if (ans >= 0.5) {
            predicted += yRight == 1;
        } else {
            predicted += yRight == 0;
        }
    }
    cout << "predicted: " << (double) predicted / testSubSet.size() * 100 << '%';
    return 0;
}
