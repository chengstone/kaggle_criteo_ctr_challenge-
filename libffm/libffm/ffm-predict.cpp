#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>
#include <memory>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <cstdlib>

#include "ffm.h"

using namespace std;
using namespace ffm;

struct Option {
    string test_path, model_path, output_path, withoutY_flag;
};

string predict_help() {
    return string(
"usage: ffm-predict test_file model_file output_file\n");
}

Option parse_option(int argc, char **argv) {
    vector<string> args;
    for(int i = 0; i < argc; i++)
        args.push_back(string(argv[i]));

    if(argc == 1)
        throw invalid_argument(predict_help());

    Option option;

    if(argc != 4 && argc != 5)
        throw invalid_argument("cannot parse argument");

    option.test_path = string(args[1]);
    option.model_path = string(args[2]);
    option.output_path = string(args[3]);
    if(argc == 5){
        option.withoutY_flag = string(args[4]);
    } else {
        option.withoutY_flag = "";
    }

    return option;
}

void predict(string test_path, string model_path, string output_path) {
    int const kMaxLineSize = 1000000;

    FILE *f_in = fopen(test_path.c_str(), "r");
    ofstream f_out(output_path);
    ofstream f_out_t(output_path + ".logit");
    char line[kMaxLineSize];

    ffm_model model = ffm_load_model(model_path);

    ffm_double loss = 0;
    vector<ffm_node> x;
    ffm_int i = 0;

    for(; fgets(line, kMaxLineSize, f_in) != nullptr; i++) {
        x.clear();
        char *y_char = strtok(line, " \t");
        ffm_float y = (atoi(y_char)>0)? 1.0f : -1.0f;

        while(true) {
            char *field_char = strtok(nullptr,":");
            char *idx_char = strtok(nullptr,":");
            char *value_char = strtok(nullptr," \t");
            if(field_char == nullptr || *field_char == '\n')
                break;

            ffm_node N;
            N.f = atoi(field_char);
            N.j = atoi(idx_char);
            N.v = atof(value_char);

            x.push_back(N);
        }

        ffm_float y_bar = ffm_predict(x.data(), x.data()+x.size(), model);
        ffm_float ret_t = ffm_get_wTx(x.data(), x.data()+x.size(), model);
        loss -= y==1? log(y_bar) : log(1-y_bar);

        f_out_t << ret_t << "\n";
        f_out << y_bar << "\n";
    }

    loss /= i;

    cout << "logloss = " << fixed << setprecision(5) << loss << endl;

    fclose(f_in);
}


void predict_withoutY(string test_path, string model_path, string output_path) {
    int const kMaxLineSize = 1000000;

    FILE *f_in = fopen(test_path.c_str(), "r");
    ofstream f_out(output_path);
    ofstream f_out_t(output_path + ".logit");
    char line[kMaxLineSize];

    ffm_model model = ffm_load_model(model_path);

    //ffm_double loss = 0;
    vector<ffm_node> x;
    ffm_int i = 0;

    for(; fgets(line, kMaxLineSize, f_in) != nullptr; i++) {
        x.clear();
        //char *y_char = strtok(line, " \t");
        //ffm_float y = (atoi(y_char)>0)? 1.0f : -1.0f;

        char *field_char = strtok(line,":");
        char *idx_char = strtok(nullptr,":");
        char *value_char = strtok(nullptr," \t");
        if(field_char == nullptr || *field_char == '\n')
            continue;

        ffm_node N;
        N.f = atoi(field_char);
        N.j = atoi(idx_char);
        N.v = atof(value_char);

        x.push_back(N);

        while(true) {
            char *field_char = strtok(nullptr,":");
            char *idx_char = strtok(nullptr,":");
            char *value_char = strtok(nullptr," \t");
            if(field_char == nullptr || *field_char == '\n')
                break;

            ffm_node N;
            N.f = atoi(field_char);
            N.j = atoi(idx_char);
            N.v = atof(value_char);

            x.push_back(N);
        }

        ffm_float y_bar = ffm_predict(x.data(), x.data()+x.size(), model);
        ffm_float ret_t = ffm_get_wTx(x.data(), x.data()+x.size(), model);
        //loss -= y==1? log(y_bar) : log(1-y_bar);

        f_out_t << ret_t << "\n";
        f_out << y_bar << "\n";
    }

    //loss /= i;

    //cout << "logloss = " << fixed << setprecision(5) << loss << endl;
    cout << "done!" << endl;

    fclose(f_in);
}

int main(int argc, char **argv) {
    Option option;
    try {
        option = parse_option(argc, argv);
    } catch(invalid_argument const &e) {
        cout << e.what() << endl;
        return 1;
    }

    if(argc == 5 && option.withoutY_flag.compare("true") == 0){
        predict_withoutY(option.test_path, option.model_path, option.output_path);
    } else {
        predict(option.test_path, option.model_path, option.output_path);
    }
    return 0;
}
