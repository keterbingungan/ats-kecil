"use client";

import { useState, useCallback } from "react";

// ROLE PRESETS (matching backend)
const ROLE_OPTIONS = [
  "Backend Engineer",
  "Frontend Engineer",
  "Full Stack Developer",
  "Data Scientist",
  "Machine Learning Engineer",
  "DevOps Engineer",
  "Mobile Developer",
  "Data Engineer",
];

// Supported file extensions
const SUPPORTED_EXTENSIONS = ['.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'];
const ACCEPTED_FILE_TYPES = "application/pdf,image/jpeg,image/png,image/bmp,image/tiff,image/webp";

// Helper to check if file is supported
const isFileSupported = (filename) => {
  const ext = '.' + filename.toLowerCase().split('.').pop();
  return SUPPORTED_EXTENSIONS.includes(ext);
};

export default function Home() {
  const [files, setFiles] = useState([]);
  const [roleName, setRoleName] = useState("Backend Engineer");
  const [additionalSkills, setAdditionalSkills] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [expandedCandidate, setExpandedCandidate] = useState(null);

  // Handle file selection
  const handleFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files).filter((f) =>
      isFileSupported(f.name)
    );
    setFiles((prev) => [...prev, ...selectedFiles]);
    setError(null);
  };

  // Handle drag and drop
  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(false);
    const droppedFiles = Array.from(e.dataTransfer.files).filter((f) =>
      isFileSupported(f.name)
    );
    setFiles((prev) => [...prev, ...droppedFiles]);
    setError(null);
  }, []);

  // Remove a file
  const removeFile = (index) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  // Clear all files
  const clearFiles = () => {
    setFiles([]);
    setResults(null);
    setError(null);
    setExpandedCandidate(null);
  };

  // Toggle candidate expansion
  const toggleExpand = (index) => {
    setExpandedCandidate(expandedCandidate === index ? null : index);
  };

  // Submit to API
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (files.length === 0) {
      setError("Please upload at least one CV (PDF)");
      return;
    }

    setIsLoading(true);
    setError(null);
    setResults(null);
    setExpandedCandidate(null);

    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));
    formData.append("role_name", roleName);
    formData.append("additional_skills", additionalSkills);

    try {
      const response = await fetch(`${API_URL}/match-cvs`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Failed to process CVs");
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err.message || "An error occurred. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Background Effects */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-cyan-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse"></div>
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-pink-500 rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-pulse"></div>
      </div>

      <div className="relative z-10 container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <header className="text-center mb-12">
          <div className="inline-flex items-center gap-3 mb-4">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-cyan-400 to-purple-600 flex items-center justify-center shadow-lg shadow-purple-500/30">
              <svg
                className="w-7 h-7 text-white"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
            </div>
            <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
              Intelligent ATS
            </h1>
          </div>
          <p className="text-slate-400 text-lg max-w-2xl mx-auto">
            AI-powered Applicant Tracking System. Upload CVs, extract skills
            automatically, and find the best candidates for your role.
          </p>
        </header>

        <div className="grid lg:grid-cols-5 gap-8">
          {/* Left Panel - Upload Form (2 cols) */}
          <div className="lg:col-span-2 space-y-6">
            {/* Upload Zone */}
            <div
              className={`
                relative rounded-2xl border-2 border-dashed transition-all duration-300 
                ${isDragOver
                  ? "border-cyan-400 bg-cyan-500/10 scale-[1.02]"
                  : "border-slate-600 hover:border-purple-500/50 bg-slate-800/50"
                }
                backdrop-blur-xl p-8
              `}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <input
                type="file"
                multiple
                accept={ACCEPTED_FILE_TYPES}
                onChange={handleFileChange}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              <div className="text-center">
                <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-br from-purple-500/20 to-cyan-500/20 flex items-center justify-center">
                  <svg
                    className={`w-8 h-8 ${isDragOver ? "text-cyan-400" : "text-purple-400"}`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                    />
                  </svg>
                </div>
                <h3 className="text-xl font-semibold text-white mb-2">
                  Drop CV Files Here
                </h3>
                <p className="text-slate-400">
                  or click to browse • PDF & images (JPG, PNG)
                </p>
              </div>
            </div>

            {/* File List */}
            {files.length > 0 && (
              <div className="bg-slate-800/50 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-white">
                    Uploaded CVs ({files.length})
                  </h3>
                  <button
                    onClick={clearFiles}
                    className="text-sm text-red-400 hover:text-red-300 transition-colors"
                  >
                    Clear All
                  </button>
                </div>
                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {files.map((file, index) => {
                    const isImage = file.name.toLowerCase().match(/\.(jpg|jpeg|png|bmp|tiff|webp)$/);
                    return (
                      <div
                        key={index}
                        className="flex items-center justify-between p-3 bg-slate-700/50 rounded-xl group"
                      >
                        <div className="flex items-center gap-3">
                          <div className={`w-10 h-10 rounded-lg ${isImage ? 'bg-blue-500/20' : 'bg-red-500/20'} flex items-center justify-center`}>
                            {isImage ? (
                              <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                              </svg>
                            ) : (
                              <svg className="w-5 h-5 text-red-400" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" clipRule="evenodd" />
                              </svg>
                            )}
                          </div>
                          <div>
                            <p className="text-sm text-white font-medium truncate max-w-[200px]">
                              {file.name}
                            </p>
                            <p className="text-xs text-slate-400">
                              {(file.size / 1024).toFixed(1)} KB • {isImage ? 'Image' : 'PDF'}
                            </p>
                          </div>
                        </div>
                        <button
                          onClick={() => removeFile(index)}
                          className="opacity-0 group-hover:opacity-100 text-slate-400 hover:text-red-400 transition-all"
                        >
                          <svg
                            className="w-5 h-5"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M6 18L18 6M6 6l12 12"
                            />
                          </svg>
                        </button>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Configuration */}
            <form
              onSubmit={handleSubmit}
              className="bg-slate-800/50 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50 space-y-5"
            >
              <h3 className="text-lg font-semibold text-white">
                Job Configuration
              </h3>

              {/* Role Selection */}
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Target Role
                </label>
                <select
                  value={roleName}
                  onChange={(e) => setRoleName(e.target.value)}
                  className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600 rounded-xl text-white focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all outline-none"
                >
                  {ROLE_OPTIONS.map((role) => (
                    <option key={role} value={role}>
                      {role}
                    </option>
                  ))}
                </select>
              </div>

              {/* Additional Skills */}
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Additional Skills (comma-separated)
                </label>
                <input
                  type="text"
                  value={additionalSkills}
                  onChange={(e) => setAdditionalSkills(e.target.value)}
                  placeholder="e.g., FastAPI, Redis, GraphQL"
                  className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600 rounded-xl text-white placeholder-slate-400 focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all outline-none"
                />
              </div>

              {/* Error Message */}
              {error && (
                <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-xl">
                  <p className="text-red-400 text-sm">{error}</p>
                </div>
              )}

              {/* Submit Button */}
              <button
                type="submit"
                disabled={isLoading || files.length === 0}
                className={`
                  w-full py-4 px-6 rounded-xl font-semibold text-white transition-all duration-300
                  ${isLoading || files.length === 0
                    ? "bg-slate-600 cursor-not-allowed"
                    : "bg-gradient-to-r from-cyan-500 to-purple-600 hover:from-cyan-400 hover:to-purple-500 shadow-lg shadow-purple-500/30 hover:shadow-purple-500/50 hover:scale-[1.02]"
                  }
                `}
              >
                {isLoading ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg
                      className="animate-spin h-5 w-5"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                        fill="none"
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      />
                    </svg>
                    Analyzing CVs...
                  </span>
                ) : (
                  `Match ${files.length} CV${files.length !== 1 ? "s" : ""}`
                )}
              </button>
            </form>
          </div>

          {/* Right Panel - Results (3 cols) */}
          <div className="lg:col-span-3 bg-slate-800/50 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50 min-h-[600px]">
            {!results ? (
              <div className="flex flex-col items-center justify-center h-full text-center">
                <div className="w-20 h-20 rounded-full bg-slate-700/50 flex items-center justify-center mb-4">
                  <svg
                    className="w-10 h-10 text-slate-500"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                    />
                  </svg>
                </div>
                <h3 className="text-xl font-semibold text-slate-400 mb-2">
                  No Results Yet
                </h3>
                <p className="text-slate-500 max-w-sm">
                  Upload CVs and click &quot;Match&quot; to see candidate rankings with
                  skill extraction analysis
                </p>
              </div>
            ) : (
              <div className="space-y-6">
                {/* Results Header */}
                <div className="flex items-center justify-between flex-wrap gap-4">
                  <div>
                    <h3 className="text-xl font-bold text-white">
                      Match Results
                    </h3>
                    <p className="text-slate-400 text-sm">
                      {results.role_name} • {results.total_candidates} candidates
                    </p>
                  </div>
                  <div className="px-3 py-1 bg-purple-500/20 rounded-full">
                    <span className="text-purple-400 text-sm font-medium">
                      Top {results.top_threshold_percent}% highlighted
                    </span>
                  </div>
                </div>

                {/* Job Skills Required */}
                <div className="bg-slate-700/30 rounded-xl p-4">
                  <h4 className="text-sm font-medium text-slate-400 mb-3">
                    Required Skills ({results.job_skills.length})
                  </h4>
                  <div className="flex flex-wrap gap-2">
                    {results.job_skills.map((skill, i) => (
                      <span
                        key={i}
                        className="px-3 py-1 bg-cyan-500/20 text-cyan-300 text-xs rounded-full border border-cyan-500/30"
                      >
                        {skill}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Candidates List */}
                <div className="space-y-4 max-h-[500px] overflow-y-auto pr-2">
                  {results.candidates.map((candidate, index) => (
                    <div
                      key={index}
                      className={`
                        rounded-xl border transition-all overflow-hidden
                        ${candidate.is_top_candidate
                          ? "bg-gradient-to-r from-purple-500/10 to-cyan-500/10 border-purple-500/30"
                          : "bg-slate-700/30 border-slate-600/30"
                        }
                      `}
                    >
                      {/* Candidate Header */}
                      <div
                        className="p-4 cursor-pointer hover:bg-white/5 transition-colors"
                        onClick={() => toggleExpand(index)}
                      >
                        <div className="flex items-start justify-between mb-3">
                          <div className="flex items-center gap-3">
                            <div
                              className={`
                                w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold
                                ${candidate.is_top_candidate
                                  ? "bg-gradient-to-br from-cyan-400 to-purple-600 text-white"
                                  : "bg-slate-600 text-slate-300"
                                }
                              `}
                            >
                              {index + 1}
                            </div>
                            <div>
                              <h4 className="text-white font-medium">
                                {candidate.filename}
                              </h4>
                              <p className="text-slate-400 text-xs">
                                {candidate.match_count}/{candidate.total_required} skills matched •
                                {candidate.extracted_skills.length} total extracted
                              </p>
                            </div>
                          </div>
                          <div className="text-right flex items-center gap-3">
                            <div>
                              <div
                                className={`
                                  text-2xl font-bold
                                  ${candidate.match_score >= 70
                                    ? "text-green-400"
                                    : candidate.match_score >= 40
                                      ? "text-yellow-400"
                                      : "text-red-400"
                                  }
                                `}
                              >
                                {candidate.match_score}%
                              </div>
                              {candidate.is_top_candidate && (
                                <span className="text-xs text-purple-400">
                                  Top Candidate
                                </span>
                              )}
                            </div>
                            <svg
                              className={`w-5 h-5 text-slate-400 transition-transform ${expandedCandidate === index ? 'rotate-180' : ''}`}
                              fill="none"
                              stroke="currentColor"
                              viewBox="0 0 24 24"
                            >
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                            </svg>
                          </div>
                        </div>

                        {/* Progress Bar */}
                        <div className="h-2 bg-slate-600 rounded-full overflow-hidden">
                          <div
                            className={`
                              h-full rounded-full transition-all duration-500
                              ${candidate.match_score >= 70
                                ? "bg-gradient-to-r from-green-500 to-emerald-400"
                                : candidate.match_score >= 40
                                  ? "bg-gradient-to-r from-yellow-500 to-orange-400"
                                  : "bg-gradient-to-r from-red-500 to-rose-400"
                              }
                            `}
                            style={{ width: `${candidate.match_score}%` }}
                          />
                        </div>

                        {/* Quick Skills Preview */}
                        <div className="mt-3 flex flex-wrap gap-1.5">
                          {candidate.matched_skills.slice(0, 5).map((skill, i) => (
                            <span
                              key={i}
                              className="px-2 py-0.5 bg-green-500/20 text-green-300 text-xs rounded border border-green-500/30"
                            >
                              ✓ {skill}
                            </span>
                          ))}
                          {candidate.matched_skills.length > 5 && (
                            <span className="px-2 py-0.5 bg-slate-600/50 text-slate-400 text-xs rounded">
                              +{candidate.matched_skills.length - 5} more
                            </span>
                          )}
                        </div>
                      </div>

                      {/* Expanded Details */}
                      {expandedCandidate === index && (
                        <div className="border-t border-slate-600/30 p-4 bg-slate-800/50 space-y-4">
                          {/* Matched Skills */}
                          <div>
                            <h5 className="text-sm font-medium text-green-400 mb-2 flex items-center gap-2">
                              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                              </svg>
                              Matched Skills ({candidate.matched_skills.length})
                            </h5>
                            <div className="flex flex-wrap gap-2">
                              {candidate.matched_skills.map((skill, i) => (
                                <span
                                  key={i}
                                  className="px-2 py-1 bg-green-500/20 text-green-300 text-xs rounded border border-green-500/30"
                                >
                                  {skill}
                                </span>
                              ))}
                              {candidate.matched_skills.length === 0 && (
                                <span className="text-slate-500 text-sm">No matching skills found</span>
                              )}
                            </div>
                          </div>

                          {/* Missing Skills */}
                          <div>
                            <h5 className="text-sm font-medium text-red-400 mb-2 flex items-center gap-2">
                              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                              </svg>
                              Missing Skills ({candidate.missing_skills.length})
                            </h5>
                            <div className="flex flex-wrap gap-2">
                              {candidate.missing_skills.map((skill, i) => (
                                <span
                                  key={i}
                                  className="px-2 py-1 bg-red-500/20 text-red-300 text-xs rounded border border-red-500/30"
                                >
                                  {skill}
                                </span>
                              ))}
                              {candidate.missing_skills.length === 0 && (
                                <span className="text-green-400 text-sm">All required skills found! 🎉</span>
                              )}
                            </div>
                          </div>

                          {/* All Extracted Skills */}
                          <div>
                            <h5 className="text-sm font-medium text-cyan-400 mb-2 flex items-center gap-2">
                              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                              </svg>
                              All Extracted Skills ({candidate.extracted_skills.length})
                            </h5>
                            <div className="flex flex-wrap gap-2 max-h-32 overflow-y-auto">
                              {candidate.extracted_skills.map((skill, i) => (
                                <span
                                  key={i}
                                  className="px-2 py-1 bg-slate-600/50 text-slate-300 text-xs rounded"
                                >
                                  {skill}
                                </span>
                              ))}
                              {candidate.extracted_skills.length === 0 && (
                                <span className="text-slate-500 text-sm">No skills extracted from CV</span>
                              )}
                            </div>
                          </div>

                          {/* Extracted Text Preview */}
                          <div>
                            <h5 className="text-sm font-medium text-slate-400 mb-2">
                              Extracted Text Preview
                            </h5>
                            <div className="bg-slate-900/50 rounded-lg p-3 max-h-32 overflow-y-auto">
                              <p className="text-xs text-slate-400 font-mono whitespace-pre-wrap">
                                {candidate.extracted_text_preview || "[No text extracted]"}
                              </p>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <footer className="text-center mt-12 text-slate-500 text-sm">
          <p>
            Powered by DistilBERT NER • Built with FastAPI & Next.js
          </p>
        </footer>
      </div>
    </div>
  );
}
