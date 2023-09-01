interface ImageViewerElementType {
  name: string
  fullPath: string
  isFolder: boolean
}

interface ImageViewerErrorType {
  error: string
  path: string
}

interface ImageViewerResultType {
  error: null | ImageViewerErrorType
  results: ImageViewerElementType[]
}
