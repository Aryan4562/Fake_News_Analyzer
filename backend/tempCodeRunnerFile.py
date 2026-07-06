class DataCollector:
    """Stage 1: Collect and validate input data"""
    
    def __init__(self):
        self.collected_data = []
        self.metadata = {}
    
    def collect_text(self, text, source="user_input"):
        """Collect text data from various sources"""
        data_point = {
            'raw_text': text,
            'source': source,
            'timestamp': self._get_timestamp(),
            'length': len(text),
            'word_count': len(text.split())
        }
        self.collected_data.append(data_point)
        return data_point
    
    def _get_timestamp(self):
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_statistics(self):
        """Get collection statistics"""
        return {
            'total_documents': len(self.collected_data),
            'avg_length': np.mean([d['length'] for d in self.collected_data]) if self.collected_data else 0,
            'sources': list(set([d['source'] for d in self.collected_data]))
        }
