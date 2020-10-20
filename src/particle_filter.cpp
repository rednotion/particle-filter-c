/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 20;  // TODO: Set the number of particles

  // generator
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  weights.resize(num_particles);
  particles.resize(num_particles);
  
  for (int i=0; i < num_particles; ++i) {
    particles[i].id = i;
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen); 
    particles[i].weight = 1;

    weights[i] = 1;
  }
  
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  
  double pred_x, pred_y, pred_theta;
  double x, y, theta;
  
  std::default_random_engine gen;
  std::normal_distribution<double> noise_x(0.0, std_pos[0]);
  std::normal_distribution<double> noise_y(0.0, std_pos[1]);
  std::normal_distribution<double> noise_theta(0.0, std_pos[2]);
  
  for (int i=0; i < num_particles; ++i) {
    x = particles[i].x;
    y = particles[i].y;
    theta = particles[i].theta;

    if (fabs(yaw_rate) > 0.0001) {
      pred_theta = theta + yaw_rate * delta_t;
      pred_x = x + (velocity / yaw_rate) * (sin(pred_theta) - sin(theta));
      pred_y = y + (velocity / yaw_rate) * (cos(theta) - cos(pred_theta));
    } else {
      pred_theta = theta;
      pred_x = x + velocity * delta_t * cos(theta);
      pred_y = y + velocity * delta_t * sin(theta);
    }
    
    //reset the particle's positions and add noise
    particles[i].x = pred_x + noise_x(gen);
    particles[i].y = pred_y + noise_y(gen);
    particles[i].theta = pred_theta + noise_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  vector<double> new_weights(num_particles);
  double stdx = std_landmark[0];
  double stdy = std_landmark[1];
  int n_obs = observations.size();
  int n_landmarks = map_landmarks.landmark_list.size();

  for (int i=0; i < num_particles; ++i) {
    double xp = particles[i].x;
    double yp = particles[i].y;
    double theta = particles[i].theta;

    // get the relevant landmarks
    vector<LandmarkObs> relevant_landmarks;
    for (int l=0; l < n_landmarks; ++l) {
      double lm_x = map_landmarks.landmark_list[l].x_f;
      double lm_y = map_landmarks.landmark_list[l].y_f;
      int lm_id = map_landmarks.landmark_list[l].id_i;
      
      // only add if it is within sensor range
      if ((fabs(lm_x - xp) <= sensor_range) && (fabs(lm_y - yp) <= sensor_range)) {
        LandmarkObs lm;
        lm.x = lm_x;
        lm.y = lm_y;
        lm.id = lm_id;
        relevant_landmarks.push_back(lm);
      }
    }
    int n_relevant_landmarks = relevant_landmarks.size();

    // for each observation
    double impt_weight = 1.0;
    vector<int> assoc_landmark_id(n_obs);
    vector<double> assoc_landmark_x(n_obs);
    vector<double> assoc_landmark_y(n_obs);

    for (int j=0; j < n_obs; ++j) {
      // convert to global map coords
      double obs_map_x = xp + observations[j].x * cos(theta) - observations[j].y * sin(theta);
      double obs_map_y = yp + observations[j].x * sin(theta) + observations[j].y * cos(theta);

      // find the best landmark to attach to it
      double min_distance = std::numeric_limits<double>::infinity();
      for (int k=0; k < n_relevant_landmarks; ++k) {
        double dist_to_landmark = dist(obs_map_x, obs_map_y, relevant_landmarks[k].x, relevant_landmarks[k].y);
        if (dist_to_landmark < min_distance) {
          min_distance = dist_to_landmark;

          assoc_landmark_x[j] = relevant_landmarks[k].x; //obs_map_x;
          assoc_landmark_y[j] = relevant_landmarks[k].y; //obs_map_y;
          assoc_landmark_id[j] = relevant_landmarks[k].id;
        }
      }
      // calculate the error
      double power_term = (pow(obs_map_x - assoc_landmark_x[j], 2)/(2*stdx*stdx)) + (pow(obs_map_y - assoc_landmark_y[j], 2)/(2*stdy*stdy));
      impt_weight *= 1 / (2 * M_PI * stdx * stdy) * exp(-power_term);
    }

    // set the details
    particles[i].weight = impt_weight;
    new_weights[i] = impt_weight;
    ParticleFilter::SetAssociations(particles[i], assoc_landmark_id, assoc_landmark_x, assoc_landmark_y);
  }
  
  // reset global weights
  weights = new_weights;
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  vector<Particle> new_particles;
  std::default_random_engine gen;
  std::discrete_distribution<long> d(weights.begin(), weights.end());
  
  for (int i=0; i < num_particles; ++i) {
    int index = d(gen);
    new_particles.push_back(particles[index]);
  }
  
  particles = new_particles;
  
// // sampling wheel implementation
//   int index = int(rand() * num_particles); // change to random;
//   float beta = 0.0;
//   float max_weight = max(weights);
//   vector<float> new_particles;

//   for (int i=0; i<num_particles; ++i) {
//     beta += rand() * 2.0 * max_weight;
//     while (beta > weights[index]) {
//       beta -= weights[index];
//       index = (index + 1) % num_particles;
//     }
//     new_particles.push_back(particles[index]);
//   }

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}