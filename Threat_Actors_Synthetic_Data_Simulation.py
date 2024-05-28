#!/usr/bin/env python
# coding: utf-8

# This is a Simulation of Threat Actors Data in Python. For Real Datasets, contact corporate@sanra.co

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Step 1: Data Collection and Preprocessing
# Load real threat actor data (for demonstration purposes, using random data)
real_data = pd.DataFrame({
    'age': np.random.randint(18, 60, 1000),
    'location': np.random.choice(['USA', 'China', 'Russia', 'Iran'], 1000),
    'motivation': np.random.choice(['financial', 'ideological', 'thrill-seeking'], 1000),
    'attack_method': np.random.choice(['phishing', 'malware', 'DDoS'], 1000),
    'personality_trait': np.random.choice(['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'], 1000)
})

# Preprocess data
scaler = StandardScaler()
real_data_scaled = scaler.fit_transform(real_data.select_dtypes(include=[np.number]))

# Step 2: Define GAN Model
def build_generator():
    model = Sequential()
    model.add(Dense(128, input_dim=100))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(real_data_scaled.shape[1], activation='tanh'))
    return model

def build_discriminator():
    model = Sequential()
    model.add(Dense(512, input_dim=real_data_scaled.shape[1]))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Compile models
optimizer = Adam(0.0002, 0.5)
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Combined model
discriminator.trainable = False
combined = Sequential([generator, discriminator])
combined.compile(loss='binary_crossentropy', optimizer=optimizer)

# Step 3: Train GAN
def train_gan(epochs, batch_size=128):
    half_batch = int(batch_size / 2)
    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, real_data_scaled.shape[0], half_batch)
        real_samples = real_data_scaled[idx]
        noise = np.random.normal(0, 1, (half_batch, 100))
        gen_samples = generator.predict(noise)
        d_loss_real = discriminator.train_on_batch(real_samples, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_samples, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        valid_y = np.array([1] * batch_size)
        g_loss = combined.train_on_batch(noise, valid_y)
        
        # Print progress
        if epoch % 100 == 0:
            print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {100*d_loss[1]}] [G loss: {g_loss}]")

# Train the GAN
train_gan(epochs=10000, batch_size=64)

# Generate synthetic data
noise = np.random.normal(0, 1, (1000, 100))
synthetic_data = generator.predict(noise)
synthetic_data = scaler.inverse_transform(synthetic_data)

# Convert to DataFrame
synthetic_data_df = pd.DataFrame(synthetic_data, columns=real_data.select_dtypes(include=[np.number]).columns)
print(synthetic_data_df.head())

