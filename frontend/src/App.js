import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Upload, FileText, BarChart3, Brain, Download, Loader2, ChevronRight, Database, Activity, TrendingUp, MessageSquare } from 'lucide-react';
import { useDropzone } from 'react-dropzone';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend, PointElement, LineElement } from 'chart.js';
import { Bar, Line } from 'react-chartjs-2';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card';
import { Button } from './components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Badge } from './components/ui/badge';
import { Alert, AlertDescription } from './components/ui/alert';
import { Progress } from './components/ui/progress';
import { ScrollArea } from './components/ui/scroll-area';
import { Separator } from './components/ui/separator';
import './App.css';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend, PointElement, LineElement);

const API_BASE_URL = process.env.REACT_APP_BACKEND_URL;

function App() {
  const [datasets, setDatasets] = useState([]);
  const [analyses, setAnalyses] = useState([]);
  const [currentAnalysis, setCurrentAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [analysisType, setAnalysisType] = useState('comprehensive');

  useEffect(() => {
    fetchDatasets();
    fetchAnalyses();
    const interval = setInterval(() => {
      if (currentAnalysis && currentAnalysis.status === 'in_progress') {
        checkAnalysisStatus(currentAnalysis.analysis_id);
      }
    }, 3000);
    return () => clearInterval(interval);
  }, [currentAnalysis]);

  const fetchDatasets = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/datasets`);
      setDatasets(response.data.datasets);
    } catch (err) {
      setError('Failed to fetch datasets');
    }
  };

  const fetchAnalyses = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/analyses`);
      setAnalyses(response.data.analyses);
    } catch (err) {
      console.error('Failed to fetch analyses:', err);
    }
  };

  const onDrop = async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setLoading(true);
    setError('');
    setUploadProgress(0);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent) => {
          const progress = (progressEvent.loaded / progressEvent.total) * 100;
          setUploadProgress(progress);
        }
      });

      await fetchDatasets();
      setSelectedDataset(response.data);
      setUploadProgress(100);
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed');
    } finally {
      setLoading(false);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls'],
      'application/json': ['.json'],
      'text/plain': ['.txt']
    },
    multiple: false
  });

  const startAnalysis = async (datasetId) => {
    setLoading(true);
    setError('');

    try {
      const response = await axios.post(`${API_BASE_URL}/api/analyze`, {
        dataset_id: datasetId,
        analysis_type: analysisType
      });

      setCurrentAnalysis(response.data);
      await fetchAnalyses();
    } catch (err) {
      setError(err.response?.data?.detail || 'Analysis failed to start');
    } finally {
      setLoading(false);
    }
  };

  const checkAnalysisStatus = async (analysisId) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/analysis/${analysisId}`);
      setCurrentAnalysis(response.data);
      
      if (response.data.status === 'completed' || response.data.status === 'failed') {
        await fetchAnalyses();
      }
    } catch (err) {
      console.error('Failed to check analysis status:', err);
    }
  };

  const downloadReport = async (analysisId, format = 'pdf') => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/export/${analysisId}?format=${format}`, {}, {
        responseType: 'blob'
      });

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `analysis_report.${format}`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError(`Failed to download ${format.toUpperCase()} report`);
    }
  };

  const renderVisualization = (key, base64Data) => {
    return (
      <div key={key} className="mb-6">
        <h4 className="text-lg font-semibold mb-3 capitalize">
          {key.replace('_', ' ')}
        </h4>
        <div className="bg-white p-4 rounded-lg border">
          <img 
            src={`data:image/png;base64,${base64Data}`} 
            alt={key}
            className="w-full h-auto max-h-96 object-contain"
          />
        </div>
      </div>
    );
  };

  const renderAnalysisResults = (results) => {
    if (!results) return null;

    return (
      <div className="space-y-6">
        {/* Executive Summary */}
        {results.report && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="w-5 h-5" />
                Executive Summary
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="prose max-w-none">
                <pre className="whitespace-pre-wrap text-sm font-sans">
                  {results.report.executive_summary}
                </pre>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Data Overview */}
        {results.data_summary && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Database className="w-5 h-5" />
                Data Overview
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
                <div className="bg-blue-50 p-4 rounded-lg">
                  <div className="text-2xl font-bold text-blue-600">
                    {results.data_summary.shape[0].toLocaleString()}
                  </div>
                  <div className="text-sm text-gray-600">Rows</div>
                </div>
                <div className="bg-green-50 p-4 rounded-lg">
                  <div className="text-2xl font-bold text-green-600">
                    {results.data_summary.shape[1]}
                  </div>
                  <div className="text-sm text-gray-600">Columns</div>
                </div>
                <div className="bg-purple-50 p-4 rounded-lg">
                  <div className="text-2xl font-bold text-purple-600">
                    {Object.values(results.data_summary.missing_values || {}).reduce((a, b) => a + b, 0)}
                  </div>
                  <div className="text-sm text-gray-600">Missing Values</div>
                </div>
                <div className="bg-orange-50 p-4 rounded-lg">
                  <div className="text-2xl font-bold text-orange-600">
                    {((results.data_summary.memory_usage || 0) / 1024 / 1024).toFixed(2)} MB
                  </div>
                  <div className="text-sm text-gray-600">Memory Usage</div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Analysis Tabs */}
        <Tabs defaultValue="quantitative" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="quantitative">Quantitative</TabsTrigger>
            <TabsTrigger value="qualitative">Qualitative</TabsTrigger>
            <TabsTrigger value="visualizations">Visualizations</TabsTrigger>
            <TabsTrigger value="report">Full Report</TabsTrigger>
          </TabsList>

          <TabsContent value="quantitative" className="mt-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="w-5 h-5" />
                  Statistical Analysis
                </CardTitle>
                <CardDescription>
                  Comprehensive statistical insights from your numerical data
                </CardDescription>
              </CardHeader>
              <CardContent>
                {results.quantitative_analysis ? (
                  <div className="space-y-4">
                    {results.quantitative_analysis.descriptive_statistics && (
                      <div>
                        <h4 className="font-semibold mb-2">Descriptive Statistics</h4>
                        <div className="overflow-x-auto">
                          <pre className="text-xs bg-gray-50 p-3 rounded">
                            {JSON.stringify(results.quantitative_analysis.descriptive_statistics, null, 2)}
                          </pre>
                        </div>
                      </div>
                    )}
                    
                    {results.quantitative_analysis.outliers && (
                      <div>
                        <h4 className="font-semibold mb-2">Outlier Analysis</h4>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                          {Object.entries(results.quantitative_analysis.outliers).map(([col, data]) => (
                            <div key={col} className="bg-red-50 p-3 rounded-lg">
                              <div className="font-medium">{col}</div>
                              <div className="text-sm text-gray-600">
                                {data.count} outliers ({data.percentage.toFixed(1)}%)
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <Alert>
                    <AlertDescription>
                      No quantitative analysis results available.
                    </AlertDescription>
                  </Alert>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="qualitative" className="mt-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <MessageSquare className="w-5 h-5" />
                  Qualitative Analysis
                </CardTitle>
                <CardDescription>
                  Text analysis, sentiment, and thematic insights
                </CardDescription>
              </CardHeader>
              <CardContent>
                {results.qualitative_analysis ? (
                  <div className="space-y-4">
                    {results.qualitative_analysis.sentiment_analysis && (
                      <div>
                        <h4 className="font-semibold mb-2">Sentiment Analysis</h4>
                        {Object.entries(results.qualitative_analysis.sentiment_analysis).map(([col, sentiment]) => (
                          <div key={col} className="mb-4 p-4 bg-gray-50 rounded-lg">
                            <div className="font-medium mb-2">{col}</div>
                            <div className="flex items-center gap-4 text-sm">
                              <div className="flex items-center gap-2">
                                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                                Positive: {sentiment.sentiment_distribution.positive}
                              </div>
                              <div className="flex items-center gap-2">
                                <div className="w-3 h-3 bg-gray-500 rounded-full"></div>
                                Neutral: {sentiment.sentiment_distribution.neutral}
                              </div>
                              <div className="flex items-center gap-2">
                                <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                                Negative: {sentiment.sentiment_distribution.negative}
                              </div>
                            </div>
                            <div className="mt-2 text-sm text-gray-600">
                              Average Sentiment: {sentiment.avg_sentiment.toFixed(3)}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ) : (
                  <Alert>
                    <AlertDescription>
                      No qualitative analysis results available.
                    </AlertDescription>
                  </Alert>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="visualizations" className="mt-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="w-5 h-5" />
                  Data Visualizations
                </CardTitle>
                <CardDescription>
                  Interactive charts and visual insights
                </CardDescription>
              </CardHeader>
              <CardContent>
                {results.visualizations && Object.keys(results.visualizations).length > 0 ? (
                  <div className="space-y-6">
                    {Object.entries(results.visualizations).map(([key, base64Data]) =>
                      renderVisualization(key, base64Data)
                    )}
                  </div>
                ) : (
                  <Alert>
                    <AlertDescription>
                      No visualizations available for this dataset.
                    </AlertDescription>
                  </Alert>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="report" className="mt-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileText className="w-5 h-5" />
                  Complete Analysis Report
                </CardTitle>
                <CardDescription>
                  Comprehensive report with findings and recommendations
                </CardDescription>
              </CardHeader>
              <CardContent>
                {results.report ? (
                  <div className="space-y-6">
                    <div className="flex gap-2 mb-4">
                      <Button
                        onClick={() => downloadReport(currentAnalysis.analysis_id, 'pdf')}
                        size="sm"
                        variant="outline"
                      >
                        <Download className="w-4 h-4 mr-2" />
                        Download PDF
                      </Button>
                      <Button
                        onClick={() => downloadReport(currentAnalysis.analysis_id, 'docx')}
                        size="sm"
                        variant="outline"
                      >
                        <Download className="w-4 h-4 mr-2" />
                        Download DOCX
                      </Button>
                    </div>

                    <div className="prose max-w-none">
                      {results.report.executive_summary && (
                        <div className="mb-6">
                          <pre className="whitespace-pre-wrap text-sm font-sans bg-blue-50 p-4 rounded-lg">
                            {results.report.executive_summary}
                          </pre>
                        </div>
                      )}

                      {results.report.detailed_findings && (
                        <div className="mb-6">
                          <pre className="whitespace-pre-wrap text-sm font-sans bg-gray-50 p-4 rounded-lg">
                            {results.report.detailed_findings}
                          </pre>
                        </div>
                      )}

                      {results.report.recommendations && (
                        <div className="mb-6">
                          <pre className="whitespace-pre-wrap text-sm font-sans bg-green-50 p-4 rounded-lg">
                            {results.report.recommendations}
                          </pre>
                        </div>
                      )}
                    </div>
                  </div>
                ) : (
                  <Alert>
                    <AlertDescription>
                      No detailed report available.
                    </AlertDescription>
                  </Alert>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header */}
      <header className="border-b border-white/10 bg-black/20 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
                <Brain className="w-7 h-7 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">DataAgent Pro</h1>
                <p className="text-purple-200 text-sm">AI-Powered Data Analysis & Intelligence</p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Badge variant="secondary" className="bg-purple-500/20 text-purple-200 border-purple-500/30">
                CrewAI Powered
              </Badge>
              <Badge variant="secondary" className="bg-blue-500/20 text-blue-200 border-blue-500/30">
                Gemini 2.0 Flash
              </Badge>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {error && (
          <Alert className="mb-6 bg-red-500/10 border-red-500/20 text-red-200">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Upload Section */}
        <Card className="mb-8 bg-white/5 border-white/10 backdrop-blur-xl">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <Upload className="w-5 h-5" />
              Upload Dataset
            </CardTitle>
            <CardDescription className="text-gray-300">
              Upload CSV, Excel, JSON, or text files for AI-powered analysis
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
                ${isDragActive 
                  ? 'border-purple-400 bg-purple-500/10' 
                  : 'border-gray-600 bg-gray-800/30 hover:bg-gray-800/50'
                }`}
            >
              <input {...getInputProps()} />
              <div className="space-y-4">
                <div className="mx-auto w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                  <Upload className="w-8 h-8 text-white" />
                </div>
                <div>
                  <p className="text-lg font-semibold text-white">
                    {isDragActive ? 'Drop your file here' : 'Drop files here or click to browse'}
                  </p>
                  <p className="text-gray-400 text-sm mt-2">
                    Supports CSV, Excel (.xlsx, .xls), JSON, and TXT files
                  </p>
                </div>
              </div>
            </div>

            {uploadProgress > 0 && uploadProgress < 100 && (
              <div className="mt-4">
                <Progress value={uploadProgress} className="w-full" />
                <p className="text-center text-sm text-gray-400 mt-2">
                  Uploading... {Math.round(uploadProgress)}%
                </p>
              </div>
            )}

            {selectedDataset && (
              <div className="mt-6 p-4 bg-green-500/10 border border-green-500/20 rounded-lg">
                <h4 className="font-semibold text-green-200 mb-2">Upload Successful!</h4>
                <div className="text-sm text-gray-300">
                  <p><strong>File:</strong> {selectedDataset.filename}</p>
                  <p><strong>Size:</strong> {selectedDataset.shape[0].toLocaleString()} rows × {selectedDataset.shape[1]} columns</p>
                </div>
                
                <div className="mt-4 flex items-center gap-4">
                  <select
                    value={analysisType}
                    onChange={(e) => setAnalysisType(e.target.value)}
                    className="bg-gray-800 border border-gray-600 rounded px-3 py-2 text-white text-sm"
                  >
                    <option value="comprehensive">Comprehensive Analysis</option>
                    <option value="quantitative">Quantitative Only</option>
                    <option value="qualitative">Qualitative Only</option>
                  </select>
                  
                  <Button
                    onClick={() => startAnalysis(selectedDataset.dataset_id)}
                    disabled={loading}
                    className="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700"
                  >
                    {loading ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Starting Analysis...
                      </>
                    ) : (
                      <>
                        <Activity className="w-4 h-4 mr-2" />
                        Start AI Analysis
                      </>
                    )}
                  </Button>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Analysis Progress */}
        {currentAnalysis && (
          <Card className="mb-8 bg-white/5 border-white/10 backdrop-blur-xl">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Activity className="w-5 h-5" />
                Analysis Status
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Badge 
                    variant={currentAnalysis.status === 'completed' ? 'default' : 'secondary'}
                    className={
                      currentAnalysis.status === 'completed' 
                        ? 'bg-green-500/20 text-green-200 border-green-500/30'
                        : currentAnalysis.status === 'in_progress'
                          ? 'bg-blue-500/20 text-blue-200 border-blue-500/30'
                          : 'bg-red-500/20 text-red-200 border-red-500/30'
                    }
                  >
                    {currentAnalysis.status === 'in_progress' && <Loader2 className="w-3 h-3 mr-1 animate-spin" />}
                    {currentAnalysis.status}
                  </Badge>
                  <span className="text-gray-300 text-sm">Analysis ID: {currentAnalysis.analysis_id.slice(0, 8)}</span>
                </div>
              </div>

              {currentAnalysis.status === 'in_progress' && (
                <div className="space-y-2">
                  <Progress value={50} className="w-full" />
                  <p className="text-center text-sm text-gray-400">
                    AI agents are analyzing your data...
                  </p>
                </div>
              )}

              {currentAnalysis.status === 'completed' && currentAnalysis.results && (
                <div className="mt-6">
                  {renderAnalysisResults(currentAnalysis.results)}
                </div>
              )}

              {currentAnalysis.error && (
                <Alert className="mt-4 bg-red-500/10 border-red-500/20 text-red-200">
                  <AlertDescription>
                    Analysis failed: {currentAnalysis.error}
                  </AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>
        )}

        {/* Recent Datasets */}
        {datasets.length > 0 && (
          <Card className="mb-8 bg-white/5 border-white/10 backdrop-blur-xl">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Database className="w-5 h-5" />
                Your Datasets
              </CardTitle>
              <CardDescription className="text-gray-300">
                Previously uploaded datasets available for analysis
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {datasets.slice(0, 5).map((dataset) => (
                  <div
                    key={dataset.dataset_id}
                    className="flex items-center justify-between p-4 bg-gray-800/50 rounded-lg border border-gray-700/50"
                  >
                    <div className="flex items-center gap-3">
                      <FileText className="w-5 h-5 text-gray-400" />
                      <div>
                        <p className="text-white font-medium">{dataset.filename}</p>
                        <p className="text-gray-400 text-sm">
                          {dataset.shape[0].toLocaleString()} rows × {dataset.shape[1]} columns
                        </p>
                      </div>
                    </div>
                    <Button
                      onClick={() => {
                        setSelectedDataset(dataset);
                        startAnalysis(dataset.dataset_id);
                      }}
                      disabled={loading}
                      size="sm"
                      variant="outline"
                      className="border-gray-600 text-gray-300 hover:bg-gray-700"
                    >
                      <ChevronRight className="w-4 h-4 mr-1" />
                      Analyze
                    </Button>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}

export default App;