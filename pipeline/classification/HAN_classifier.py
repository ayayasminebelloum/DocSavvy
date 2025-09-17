import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Bidirectional, GRU, Dense, TimeDistributed
from tensorflow.keras.layers import Dropout, Attention, Lambda, Layer, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import spacy
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pathlib import Path
import pickle
import joblib
from typing import List, Dict, Tuple, Optional, Union

# Ensure TensorFlow limits GPU memory usage
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Custom Attention Layer
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        # Alignment scores. Shape: (batch_size, seq_len, 1)
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        
        # Attention weights. Shape: (batch_size, seq_len, 1)
        a = tf.nn.softmax(e, axis=1)
        
        # Weighted sum. Shape: (batch_size, hidden_size)
        output = tf.matmul(tf.transpose(x, perm=[0, 2, 1]), a)
        return tf.squeeze(output, axis=-1)
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]
    
    def get_config(self):
        return super(AttentionLayer, self).get_config()


class HierarchicalAttentionNetwork:
    """
    Hierarchical Attention Network for Document Classification
    
    This model implements the architecture from the paper:
    "Hierarchical Attention Networks for Document Classification"
    by Yang et al., 2016
    
    It uses a hierarchical structure with attention at both the word and sentence levels.
    """
    def __init__(self, 
                 max_features: int = 20000,
                 max_sentences: int = 15,
                 max_sentence_length: int = 50,
                 embedding_dim: int = 100,
                 gru_units: int = 100,
                 dropout_rate: float = 0.3,
                 learning_rate: float = 0.001,
                 model_dir: str = "../models"):
        """
        Initialize the HAN model
        
        Parameters:
        -----------
        max_features : int
            Size of vocabulary
        max_sentences : int
            Maximum number of sentences per document
        max_sentence_length : int
            Maximum number of words per sentence
        embedding_dim : int
            Dimension of word embeddings
        gru_units : int 
            Units in GRU layer
        dropout_rate : float
            Dropout rate for regularization
        learning_rate : float
            Learning rate for Adam optimizer
        model_dir : str
            Directory to save models
        """
        self.max_features = max_features
        self.max_sentences = max_sentences
        self.max_sentence_length = max_sentence_length
        self.embedding_dim = embedding_dim
        self.gru_units = gru_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize tokenizer
        self.tokenizer = Tokenizer(num_words=max_features)
        
        # Initialize spaCy for text preprocessing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Download if model isn't available
            import subprocess
            print(f"Downloading spaCy model en_core_web_sm...")
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            self.nlp = spacy.load("en_core_web_sm")
            
        # Set attributes that will be initialized during training
        self.model = None
        self.word_encoder = None
        self.sentence_encoder = None
        self.label_map = None
        self.class_weights = None
        self.n_classes = None
        
    def preprocess_document(self, text: str) -> List[List[str]]:
        """
        Preprocess document into sentences and words
        
        Parameters:
        -----------
        text : str
            Document text
            
        Returns:
        --------
        List[List[str]]
            List of sentences, each containing a list of words
        """
        # Handle missing or invalid input
        if not isinstance(text, str) or pd.isna(text):
            return [[""]]
        
        # Clean text
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        text = re.sub(r'[^\w\s\.,\-\:/\?\(\)]', '', text)  # Keep basic punctuation
        
        # Process with spaCy
        doc = self.nlp(text)
        
        # Split into sentences
        sentences = []
        for sent in doc.sents:
            # Extract and clean tokens
            words = [token.lemma_.lower() for token in sent 
                     if not token.is_punct and token.text.strip()]
            
            # Add special tokens for document structure (helps classification)
            sent_text = sent.text.lower()
            if re.search(r'invoice|bill|receipt', sent_text):
                words.append("DOC_TRANSACTION")
            if re.search(r'total|amount|sum|paid', sent_text):
                words.append("DOC_FINANCIAL")
            if re.search(r'@|email|sent|from|to:', sent_text):
                words.append("DOC_EMAIL")
            if re.search(r'dear|hello|hi|regards|sincerely', sent_text):
                words.append("DOC_CORRESPONDENCE")
                
            if words:  # Only add non-empty sentences
                sentences.append(words)
        
        # Ensure we have at least one sentence
        if not sentences:
            sentences = [["UNK"]]
            
        return sentences
    
    def prepare_sequences(self, texts: List[str], fit_tokenizer: bool = True) -> np.ndarray:
        """
        Convert documents to hierarchical sequences for the HAN model
        
        Parameters:
        -----------
        texts : List[str]
            List of document texts
        fit_tokenizer : bool
            Whether to fit the tokenizer on these texts
            
        Returns:
        --------
        np.ndarray
            3D array with shape (n_docs, max_sentences, max_sentence_length)
        """
        # Preprocess each document
        processed_docs = [self.preprocess_document(text) for text in texts]
        
        # Flatten all words for tokenizer
        all_words = []
        for doc in processed_docs:
            for sentence in doc:
                all_words.extend(sentence)
        
        # Fit or use tokenizer
        if fit_tokenizer:
            self.tokenizer.fit_on_texts(all_words)
            
            # Save tokenizer
            tokenizer_path = self.model_dir / "han_tokenizer.pickle"
            with open(tokenizer_path, 'wb') as handle:
                pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Convert each sentence to sequence
        docs_sequences = np.zeros((len(processed_docs), self.max_sentences, self.max_sentence_length), dtype='int32')
        
        for i, doc in enumerate(processed_docs):
            for j, sentence in enumerate(doc):
                if j < self.max_sentences:
                    # Convert sentence to sequence
                    sequence = self.tokenizer.texts_to_sequences([sentence])[0]
                    # Pad sequence
                    if len(sequence) > self.max_sentence_length:
                        sequence = sequence[:self.max_sentence_length]
                    
                    # Store in array
                    docs_sequences[i, j, :len(sequence)] = sequence
        
        return docs_sequences
    
    def build_model(self, n_classes: int):
        """
        Build the Hierarchical Attention Network model
        
        Parameters:
        -----------
        n_classes : int
            Number of classes for classification
        """
        # Word-level attention model
        word_input = Input(shape=(self.max_sentence_length,), name='word_input')
        word_embedding = Embedding(input_dim=self.max_features, 
                                  output_dim=self.embedding_dim,
                                  input_length=self.max_sentence_length,
                                  name='word_embedding')(word_input)
        word_gru = Bidirectional(GRU(self.gru_units, return_sequences=True, name='word_gru'))(word_embedding)
        word_attention = AttentionLayer(name='word_attention')(word_gru)
        word_dropout = Dropout(self.dropout_rate)(word_attention)
        
        # Word Encoder model
        self.word_encoder = Model(word_input, word_dropout)
        
        # Sentence-level attention model
        sentence_input = Input(shape=(self.max_sentences, self.max_sentence_length), name='sentence_input')
        
        # Apply word encoder to each sentence
        sentence_encoder = TimeDistributed(self.word_encoder)(sentence_input)
        
        # Sentence-level GRU and attention
        sentence_gru = Bidirectional(GRU(self.gru_units, return_sequences=True, name='sentence_gru'))(sentence_encoder)
        sentence_attention = AttentionLayer(name='sentence_attention')(sentence_gru)
        sentence_dropout = Dropout(self.dropout_rate)(sentence_attention)
        
        # Output layer
        output = Dense(n_classes, activation='softmax', name='output')(sentence_dropout)
        
        # Full model
        self.model = Model(sentence_input, output)
        
        # Compile model
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=['accuracy']
        )
        
        # Print model summary
        print(self.model.summary())
    
    def train(self, train_path: str, validation_path: str = None, validation_split: float = 0.1, 
              batch_size: int = 16, epochs: int = 20):
        """
        Train the HAN model
        
        Parameters:
        -----------
        train_path : str
            Path to training data CSV
        validation_path : str, optional
            Path to validation data CSV. If provided, validation_split is ignored.
        validation_split : float
            Fraction of training data to use for validation (only used if validation_path is None)
        batch_size : int
            Batch size for training
        epochs : int
            Maximum epochs for training
        """
        # Load training data
        train_df = pd.read_csv(train_path)
        train_df['text'] = train_df['text'].fillna("")
        
        print(f"Training on {len(train_df)} documents")
        
        # Get class distribution
        class_distribution = train_df['label'].value_counts().sort_index()
        print(f"\nClass distribution:")
        print(class_distribution)
        
        # Prepare label mapping and weights
        unique_labels = sorted(train_df['label'].unique())
        self.n_classes = len(unique_labels)
        self.label_map = {label: i for i, label in enumerate(unique_labels)}
        
        # Calculate class weights to handle imbalance
        total_samples = len(train_df)
        self.class_weights = {i: total_samples / (len(unique_labels) * count) 
                             for i, (label, count) in enumerate(class_distribution.items())}
        
        # Convert labels to one-hot encoding
        labels = np.array([self.label_map[label] for label in train_df['label']])
        y_train = to_categorical(labels, num_classes=self.n_classes)
        
        # Prepare document sequences
        print("\nPreprocessing documents...")
        X_train = self.prepare_sequences(train_df['text'].tolist(), fit_tokenizer=True)
        print(f"Document sequence shape: {X_train.shape}")
        
        # Build model if not already built
        if self.model is None:
            print("\nBuilding HAN model...")
            self.build_model(self.n_classes)
        
        # Handle validation data
        if validation_path:
            # Use separate validation file
            print(f"Using separate validation data from {validation_path}")
            val_df = pd.read_csv(validation_path)
            val_df['text'] = val_df['text'].fillna("")
            
            # Prepare validation sequences
            X_val = self.prepare_sequences(val_df['text'].tolist(), fit_tokenizer=False)
            
            # Convert validation labels to one-hot
            val_labels = np.array([self.label_map.get(label, 0) for label in val_df['label']])
            y_val = to_categorical(val_labels, num_classes=self.n_classes)
            
            print(f"Validation data shape: {X_val.shape}")
        else:
            # Split into training and validation sets
            print(f"Splitting training data with validation_split={validation_split}")
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=validation_split, random_state=42, stratify=labels)
        
        # Set up callbacks
        checkpoint_path = self.model_dir / "han_model_best.weights.h5"
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint(filepath=str(checkpoint_path), 
                           monitor='val_loss', 
                           save_best_only=True,
                           save_weights_only=True)
        ]
        
        # Train model
        print("\nTraining model...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=self.class_weights,
            callbacks=callbacks
        )
        
        # Save model configuration
        config = {
            'max_features': self.max_features,
            'max_sentences': self.max_sentences,
            'max_sentence_length': self.max_sentence_length,
            'embedding_dim': self.embedding_dim,
            'gru_units': self.gru_units,
            'dropout_rate': self.dropout_rate,
            'label_map': self.label_map,
            'class_weights': self.class_weights,
            'n_classes': self.n_classes
        }
        
        config_path = self.model_dir / "han_config.joblib"
        joblib.dump(config, config_path)
        
        # Save final model weights
        model_path = self.model_dir / "han_model_final.weights.h5"
        self.model.save_weights(str(model_path))
        
        print(f"Model trained and saved to {model_path}")
        return history
    
    def evaluate(self, test_path: str, batch_size: int = 16):
        """
        Evaluate the model on test data
        
        Parameters:
        -----------
        test_path : str
            Path to test data CSV
        batch_size : int
            Batch size for evaluation
            
        Returns:
        --------
        dict
            Evaluation metrics
        """
        # Ensure model is built
        if self.model is None:
            raise ValueError("Model has not been built or trained yet")
        
        # Load test data
        test_df = pd.read_csv(test_path)
        test_df['text'] = test_df['text'].fillna("")
        
        print(f"Evaluating on {len(test_df)} documents")
        
        # Prepare test sequences
        X_test = self.prepare_sequences(test_df['text'].tolist(), fit_tokenizer=False)
        
        # Convert labels to one-hot encoding
        labels = np.array([self.label_map.get(label, 0) for label in test_df['label']])
        y_test = to_categorical(labels, num_classes=self.n_classes)
        
        # Evaluate model
        metrics = self.model.evaluate(X_test, y_test, batch_size=batch_size)
        print(f"\nTest loss: {metrics[0]:.4f}")
        print(f"Test accuracy: {metrics[1]:.4f}")
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test, batch_size=batch_size)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Map back to original labels
        reverse_label_map = {v: k for k, v in self.label_map.items()}
        original_labels = np.array([reverse_label_map[i] for i in y_pred])
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(test_df['label'], original_labels))
        
        return {
            'loss': metrics[0],
            'accuracy': metrics[1],
            'predictions': original_labels
        }
    
    def predict(self, texts: Union[str, List[str]]):
        """
        Predict document class(es)
        
        Parameters:
        -----------
        texts : str or List[str]
            Document text(s) to classify
            
        Returns:
        --------
        np.ndarray
            Predicted class label(s)
        """
        # Handle single text
        single_text = False
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        
        # Ensure model is built
        if self.model is None:
            raise ValueError("Model has not been built or trained yet")
        
        # Prepare sequences
        X_pred = self.prepare_sequences(texts, fit_tokenizer=False)
        
        # Make predictions
        y_pred_proba = self.model.predict(X_pred)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Map back to original labels
        reverse_label_map = {v: k for k, v in self.label_map.items()}
        original_labels = np.array([reverse_label_map[i] for i in y_pred])
        
        # Return single value if input was single text
        if single_text:
            return original_labels[0]
        
        return original_labels
    
    def load_model(self, model_path: Optional[str] = None, config_path: Optional[str] = None):
        """
        Load a trained model
        
        Parameters:
        -----------
        model_path : str, optional
            Path to model weights (.weights.h5 file)
        config_path : str, optional
            Path to model configuration
        """
        # Default paths
        if model_path is None:
            model_path = self.model_dir / "han_model_final.weights.h5"
        if config_path is None:
            config_path = self.model_dir / "han_config.joblib"
        
        model_path = Path(model_path)
        config_path = Path(config_path)
            
        # Load configuration
        config = joblib.load(config_path)
        
        # Update attributes
        for key, value in config.items():
            setattr(self, key, value)
        
        # Load tokenizer
        tokenizer_path = self.model_dir / "han_tokenizer.pickle"
        with open(tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        
        # Build model
        self.build_model(self.n_classes)
        
        # Load weights
        self.model.load_weights(str(model_path))
        
        print(f"Model loaded from {model_path}")
        return self
    
    def extract_attention_weights(self, text: str):
        """
        Extract attention weights for visualization
        
        Parameters:
        -----------
        text : str
            Document text
            
        Returns:
        --------
        dict
            Attention weights at word and sentence levels
        """
        # Ensure model is built
        if self.model is None or self.word_encoder is None:
            raise ValueError("Model has not been built or trained yet")
        
        # Preprocess document
        processed_doc = [self.preprocess_document(text)]
        
        # Convert to sequence
        X = self.prepare_sequences([text], fit_tokenizer=False)
        
        # Get word attention model
        word_attention_layer = self.word_encoder.get_layer('word_attention')
        
        # Create a model that outputs attention weights
        word_attention_model = Model(
            inputs=self.word_encoder.input,
            outputs=word_attention_layer.output
        )
        
        # Get sentence attention model
        sentence_attention_layer = self.model.get_layer('sentence_attention')
        
        # Create a model that outputs attention weights
        sentence_attention_model = Model(
            inputs=self.model.input,
            outputs=sentence_attention_layer.output
        )
        
        # Get attention weights
        word_attention_weights = []
        for i in range(min(len(processed_doc[0]), self.max_sentences)):
            sentence_seq = X[0, i, :]
            if np.sum(sentence_seq) > 0:  # Only process non-empty sentences
                sentence_input = np.expand_dims(sentence_seq, axis=0)
                # Get word attention weights
                word_weights = word_attention_model.predict(sentence_input)
                word_attention_weights.append(word_weights)
        
        # Get sentence attention weights
        sentence_attention_weights = sentence_attention_model.predict(X)
        
        return {
            'word_attention': word_attention_weights,
            'sentence_attention': sentence_attention_weights,
            'processed_doc': processed_doc[0]
        }


# Example usage
if __name__ == "__main__":
    # Initialize model
    han_model = HierarchicalAttentionNetwork(
        max_features=20000,
        max_sentences=15,
        max_sentence_length=50,
        embedding_dim=100,
        gru_units=100
    )
    
    # Train model
    han_model.train("../data/train.csv", epochs=10)
    
    # Evaluate model
    eval_metrics = han_model.evaluate("../data/test.csv")
    
    # Make prediction on new text
    new_text = "Invoice #12345 Customer: John Doe Amount: $100.00 Due Date: 2023-05-01"
    pred_label = han_model.predict(new_text)
    print(f"Predicted class: {pred_label}")