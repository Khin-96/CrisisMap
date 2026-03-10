# CrisisMap v2.0 - Silicon Valley Grade Platform

## Advanced Conflict Early Warning & Predictive Analytics

CrisisMap v2.0 is a complete transformation from prototype to production-ready platform, featuring modern architecture, interactive visualizations, and enterprise-grade capabilities.

## New Features

### Modern Frontend
- **Next.js 14** with TypeScript for production performance
- **shadcn/ui** components for consistent design system
- **Framer Motion** animations for warnings and model evaluations
- **Three.js** 3D globe visualizations for conflict mapping
- **Real-time WebSocket** updates for live crisis monitoring
- **Progressive Web App** capabilities for offline access

### Advanced ML Pipeline
- **MongoDB** for flexible, scalable data storage
- **CSV Upload Interface** with drag-and-drop functionality
- **Automated Model Training** with hyperparameter optimization
- **Model Versioning** and performance tracking
- **Real-time Predictions** with confidence intervals
- **Anomaly Detection** for unusual conflict patterns

### Enterprise Architecture
- **Microservices** with Docker containerization
- **Horizontal Scaling** with load balancing
- **API Gateway** with rate limiting and authentication
- **Real-time Analytics** with streaming data processing
- **Comprehensive Monitoring** with health checks and metrics

## Technology Stack

### Frontend
- **Framework**: Next.js 14 with TypeScript
- **UI Library**: shadcn/ui + Tailwind CSS
- **Animations**: Framer Motion
- **3D Graphics**: Three.js + React Three Fiber
- **State Management**: Zustand
- **Data Fetching**: TanStack Query
- **Charts**: Recharts + D3.js

### Backend
- **API**: FastAPI with async/await
- **Database**: MongoDB with Motor (async driver)
- **Caching**: Redis for performance
- **ML**: scikit-learn, XGBoost, LightGBM
- **WebSockets**: Real-time communication
- **File Processing**: Pandas, NumPy

### Infrastructure
- **Containerization**: Docker + Docker Compose
- **Reverse Proxy**: Nginx
- **Monitoring**: Prometheus + Grafana
- **CI/CD**: GitHub Actions
- **Cloud**: AWS/GCP/Azure ready

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Node.js 18+
- 8GB RAM minimum
- 20GB disk space

### Installation

1. **Clone and Setup (Windows)**
   ```cmd
   git clone <repository>
   cd crisismap
   setup_v2.bat
   ```

2. **Clone and Setup (Linux/Mac)**
   ```bash
   git clone <repository>
   cd crisismap
   chmod +x setup_v2.sh
   ./setup_v2.sh
   ```

3. **Access the Platform**
   - Frontend: http://localhost:3000
   - API: http://localhost:8000
   - Docs: http://localhost:8000/docs

### Development Mode

1. **Start Backend**
   ```bash
   cd backend_v2
   pip install -r requirements_v2.txt
   uvicorn main:app --reload
   ```

2. **Start Frontend**
   ```bash
   cd next-frontend
   npm install
   npm run dev
   ```

## Key Capabilities

### Interactive Visualizations
- **3D Globe**: Real-time conflict event mapping with particle effects
- **Animated Charts**: Smooth transitions for temporal trends
- **Heatmaps**: Geographic intensity with smooth color transitions
- **Network Graphs**: Actor relationship visualization
- **Timeline**: Conflict escalation pattern analysis

### Machine Learning
- **Predictive Models**: Random Forest, Gradient Boosting, XGBoost
- **Feature Engineering**: Temporal, geographic, and contextual features
- **Model Evaluation**: Cross-validation, drift detection, A/B testing
- **Hyperparameter Tuning**: Automated optimization with Optuna
- **Explainability**: SHAP values and feature importance

### Advanced Analytics
- **Anomaly Detection**: Statistical and ML-based outlier identification
- **Trend Analysis**: Seasonal decomposition and forecasting
- **Hotspot Identification**: Geographic clustering algorithms
- **Risk Assessment**: Multi-factor scoring and alerting
- **Impact Analysis**: Scenario modeling and simulation

### Data Pipeline
- **CSV Upload**: Drag-and-drop with validation and preview
- **Data Quality**: Automated cleaning and validation
- **Schema Detection**: Intelligent column mapping
- **Batch Processing**: Chunked file handling for large datasets
- **Real-time Ingestion**: Streaming data from external APIs

## Enterprise Features

### Security & Compliance
- **Authentication**: JWT tokens, OAuth2, SSO integration
- **Authorization**: Role-based access control (RBAC)
- **Data Encryption**: At-rest and in-transit encryption
- **Audit Logging**: Comprehensive activity tracking
- **GDPR Compliance**: Data privacy and retention policies

### Monitoring & Observability
- **Health Checks**: Service availability monitoring
- **Performance Metrics**: Response times, throughput, errors
- **Business Metrics**: User engagement, data quality, model performance
- **Alerting**: Slack, email, PagerDuty integrations
- **Dashboards**: Real-time operational visibility

### Scalability & Performance
- **Horizontal Scaling**: Auto-scaling based on load
- **Caching**: Multi-layer caching strategy
- **CDN**: Global content delivery
- **Database Optimization**: Indexing, sharding, read replicas
- **API Rate Limiting**: Prevent abuse and ensure fair usage

## Project Structure

```
crisismap/
├── next-frontend/        # Next.js frontend application
│   ├── src/
│   │   ├── app/         # App router pages
│   │   ├── components/  # React components
│   │   ├── lib/        # Utilities and API client
│   │   └── hooks/      # Custom React hooks
│   ├── package.json    # Frontend dependencies
│   ├── next.config.js  # Next.js configuration
│   └── tailwind.config.ts # Tailwind CSS config
├── backend_v2/          # FastAPI backend
│   ├── main.py         # API server
│   ├── database.py     # MongoDB manager
│   ├── ml_pipeline.py  # ML training and inference
│   ├── data_processor.py # Data processing utilities
│   └── websocket_manager.py # Real-time communication
├── docker-compose.yml   # Container orchestration
├── Dockerfile.backend   # Backend container
├── Dockerfile.frontend  # Frontend container
├── setup_v2.sh         # Linux/Mac setup script
├── setup_v2.bat        # Windows setup script
└── nginx.conf          # Reverse proxy configuration
```

## Configuration

### Environment Variables
```bash
# Database
MONGODB_URL=mongodb://admin:password@localhost:27017/crisismap
REDIS_URL=redis://localhost:6379

# API
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000

# Security
JWT_SECRET_KEY=your-secret-key
ENCRYPTION_KEY=your-encryption-key

# External APIs
ACLED_API_KEY=your-acled-key
HDX_API_KEY=your-hdx-key
```

### Database Configuration
- **MongoDB**: Document-based storage for flexible schema
- **Indexes**: Optimized for geospatial and temporal queries
- **Sharding**: Horizontal partitioning for scale
- **Replication**: High availability and disaster recovery

## Testing & Quality

### Testing Strategy
- **Unit Tests**: Component and function testing
- **Integration Tests**: API and database testing
- **E2E Tests**: Full user workflow testing
- **Performance Tests**: Load and stress testing
- **Security Tests**: Vulnerability scanning

### Code Quality
- **TypeScript**: Static type checking
- **ESLint**: Code linting and formatting
- **Prettier**: Code formatting
- **Husky**: Git hooks for quality gates
- **SonarQube**: Code quality analysis

## Deployment

### Production Deployment
1. **Cloud Infrastructure**: AWS/GCP/Azure
2. **Container Orchestration**: Kubernetes or ECS
3. **Database**: Managed MongoDB Atlas
4. **Caching**: Managed Redis
5. **CDN**: CloudFront or CloudFlare
6. **Monitoring**: DataDog or New Relic

### CI/CD Pipeline
1. **Source Control**: Git with feature branches
2. **Build**: Docker image creation
3. **Test**: Automated test suite
4. **Security**: Vulnerability scanning
5. **Deploy**: Blue-green deployment
6. **Monitor**: Health checks and rollback

## API Documentation

### Core Endpoints
- `GET /api/events` - Retrieve conflict events
- `GET /api/trends` - Get temporal trends and predictions
- `POST /api/upload/csv` - Upload training data
- `POST /api/ml/train` - Train ML models
- `GET /api/analytics/anomalies` - Detect anomalies
- `WebSocket /ws` - Real-time updates

### Authentication
```bash
# Get access token
curl -X POST /api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass"}'

# Use token in requests
curl -H "Authorization: Bearer <token>" /api/events
```

## Contributing

1. **Fork** the repository
2. **Create** a feature branch
3. **Commit** your changes
4. **Push** to the branch
5. **Create** a Pull Request

### Development Guidelines
- Follow TypeScript best practices
- Write comprehensive tests
- Update documentation
- Follow semantic versioning
- Use conventional commits

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

- **Documentation**: [docs.crisismap.io](https://docs.crisismap.io)
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: support@crisismap.io

---

**CrisisMap v2.0** - Transforming conflict analysis with Silicon Valley-grade technology.