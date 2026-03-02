require 'json'

package = JSON.parse(File.read(File.join(File.dirname(__FILE__), 'package.json')))

# Xcode 26 beta workaround: duplicate .modulemap files in the SDK cause
# "redefinition of module" errors with explicit modules enabled.
xcode_major = begin
  m = `xcrun xcodebuild -version 2>/dev/null`.to_s.match(/Xcode (\d+)/)
  m ? m[1].to_i : 0
rescue
  0
end

Pod::Spec.new do |s|
  s.name = 'DustEmbeddingsCapacitor'
  s.version = package['version']
  s.summary = package['description']
  s.license = package['license']
  s.homepage = 'https://github.com/rogelioRuiz/dust-embeddings-capacitor'
  s.author = 'Rogelio Ruiz Perez'
  s.source = { :git => 'https://github.com/rogelioRuiz/dust-embeddings-capacitor.git', :tag => s.version.to_s }

  s.source_files = 'ios/Sources/EmbeddingsPlugin/EmbeddingsPlugin.swift'
  s.ios.deployment_target = '16.0'

  s.dependency 'Capacitor'
  s.dependency 'DustCoreCapacitor'
  s.dependency 'DustCore'
  s.dependency 'DustEmbeddings'
  s.dependency 'DustOnnx'
  s.swift_version = '5.9'

  if xcode_major >= 26
    s.pod_target_xcconfig = { 'SWIFT_ENABLE_EXPLICIT_MODULES' => 'NO' }
  end
end
