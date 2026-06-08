// =============================================================================
// RAG-pattern-after-Build demo — Azure infrastructure
// -----------------------------------------------------------------------------
// Provisions:
//   • Azure AI Search (Standard, system-assigned MSI, semantic ranker on,
//     local auth disabled) — supports agentic retrieval REST API 2026-04-01.
//   • Azure OpenAI account with two deployments:
//        text-embedding-3-large  (for the integrated AzureOpenAIVectorizer)
//        gpt-5-mini              (for agentic synthesis)
//   • RBAC for keyless auth via DefaultAzureCredential:
//        – deploying user → Search Service Contributor + Search Index Data
//          Contributor on the search service.
//        – deploying user → Cognitive Services OpenAI User on the AOAI acct.
//        – search service MSI → Cognitive Services OpenAI User on the AOAI
//          acct (so the integrated vectorizer + knowledge agent can call the
//          embedding & chat models without a key).
// =============================================================================

targetScope = 'resourceGroup'

// -----------------------------------------------------------------------------
// Parameters
// -----------------------------------------------------------------------------

@description('Azure region for all resources. Default: eastus2.')
param location string = 'eastus2'

@description('Region for the Azure AI Search service. Override if the primary location lacks Search capacity. Default: same as location.')
param searchLocation string = location

@description('Short prefix used to name resources. 2–11 lowercase chars.')
@minLength(2)
@maxLength(11)
param baseName string = 'ragbuild'

@description('Object ID of the user / service principal that will run the demo. Get with: az ad signed-in-user show --query id -o tsv')
param principalId string

@description('Principal type for the deploying identity.')
@allowed([
  'User'
  'ServicePrincipal'
  'Group'
])
param principalType string = 'User'

@description('Azure AI Search SKU. standard required for agentic retrieval + semantic ranker.')
@allowed([
  'standard'
  'standard2'
  'standard3'
])
param searchSku string = 'standard'

@description('Capacity (K tokens / minute) for the gpt-5-mini deployment.')
param chatCapacity int = 50

@description('Capacity (K tokens / minute) for the text-embedding-3-large deployment.')
param embeddingCapacity int = 50

@description('Container image for the Streamlit web app. azd swaps this on `azd deploy`.')
param webImage string = 'mcr.microsoft.com/k8se/quickstart:latest'

@description('vCPU per Container App replica.')
param webCpu string = '0.5'

@description('Memory per Container App replica (e.g. 1.0Gi).')
param webMemory string = '1.0Gi'

@description('Min replicas (set 0 for scale-to-zero, 1 to avoid cold start).')
@minValue(0)
param webMinReplicas int = 1

@description('Max replicas for the Streamlit container.')
@minValue(1)
param webMaxReplicas int = 3

@description('Tags applied to all resources (azd injects azd-env-name here).')
param tags object = {}

// -----------------------------------------------------------------------------
// Variables — deterministic names & built-in role definition IDs
// -----------------------------------------------------------------------------

var suffix = uniqueString(resourceGroup().id)
var searchName = toLower('srch-${baseName}-${suffix}')
var openaiName = toLower('oai-${baseName}-${suffix}')
var uamiName = toLower('id-${baseName}-${suffix}')
var lawName = toLower('log-${baseName}-${suffix}')
var acaEnvName = toLower('cae-${baseName}-${suffix}')
var acrName = toLower('acr${baseName}${suffix}') // ACR names: alphanumeric only
var webAppName = toLower('ca-${baseName}-web')

var chatDeploymentName = 'gpt-5-mini'
var chatModelName = 'gpt-5-mini'
var chatModelVersion = '2025-08-07'
var embeddingDeploymentName = 'text-embedding-3-large'
var embeddingModelName = 'text-embedding-3-large'
var embeddingModelVersion = '1'

// Built-in role definitions (resource-id form)
var roleSearchServiceContributor = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7ca78c08-252a-4471-8644-bb5ff32d4ba0')
var roleSearchIndexDataContributor = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '8ebe5a00-799e-43f5-93ac-243d3dce84a7')
var roleSearchIndexDataReader = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '1407120a-92aa-4202-b7e9-c0e197c71c8f')
var roleCognitiveServicesOpenAIUser = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd')
var roleAcrPull = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')

// -----------------------------------------------------------------------------
// Azure OpenAI account + model deployments
// -----------------------------------------------------------------------------

resource openai 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: openaiName
  location: location
  kind: 'OpenAI'
  sku: {
    name: 'S0'
  }
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    customSubDomainName: openaiName
    publicNetworkAccess: 'Enabled'
    disableLocalAuth: true
    networkAcls: {
      defaultAction: 'Allow'
    }
  }
}

resource embeddingDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: openai
  name: embeddingDeploymentName
  sku: {
    name: 'Standard'
    capacity: embeddingCapacity
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: embeddingModelName
      version: embeddingModelVersion
    }
  }
}

resource chatDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: openai
  name: chatDeploymentName
  sku: {
    name: 'GlobalStandard'
    capacity: chatCapacity
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: chatModelName
      version: chatModelVersion
    }
  }
  dependsOn: [
    embeddingDeployment
  ]
}

// -----------------------------------------------------------------------------
// Azure AI Search
// -----------------------------------------------------------------------------

resource search 'Microsoft.Search/searchServices@2024-03-01-preview' = {
  name: searchName
  location: searchLocation
  sku: {
    name: searchSku
  }
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    replicaCount: 1
    partitionCount: 1
    hostingMode: 'default'
    publicNetworkAccess: 'enabled'
    disableLocalAuth: true
    semanticSearch: 'standard'
    authOptions: null
  }
}

// -----------------------------------------------------------------------------
// Role assignments
// -----------------------------------------------------------------------------

// Deploying user → Search Service Contributor (create/manage indexes, knowledge sources, knowledge bases)
resource raUserSearchSvc 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: search
  name: guid(search.id, principalId, roleSearchServiceContributor)
  properties: {
    principalId: principalId
    principalType: principalType
    roleDefinitionId: roleSearchServiceContributor
  }
}

// Deploying user → Search Index Data Contributor (upload docs, query data plane)
resource raUserSearchData 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: search
  name: guid(search.id, principalId, roleSearchIndexDataContributor)
  properties: {
    principalId: principalId
    principalType: principalType
    roleDefinitionId: roleSearchIndexDataContributor
  }
}

// Deploying user → Cognitive Services OpenAI User (call AOAI from app + lab scripts)
resource raUserOpenAI 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: openai
  name: guid(openai.id, principalId, roleCognitiveServicesOpenAIUser)
  properties: {
    principalId: principalId
    principalType: principalType
    roleDefinitionId: roleCognitiveServicesOpenAIUser
  }
}

// Search service MSI → Cognitive Services OpenAI User (integrated vectorizer + knowledge agent)
resource raSearchOpenAI 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: openai
  name: guid(openai.id, search.id, roleCognitiveServicesOpenAIUser)
  properties: {
    principalId: search.identity.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: roleCognitiveServicesOpenAIUser
  }
}

// -----------------------------------------------------------------------------
// Web app hosting: UAMI + Log Analytics + ACR + Container Apps env + Container App
// -----------------------------------------------------------------------------

resource uami 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: uamiName
  location: location
  tags: tags
}

resource law 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: lawName
  location: location
  tags: tags
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
    features: {
      enableLogAccessUsingOnlyResourcePermissions: true
    }
  }
}

resource acaEnv 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: acaEnvName
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: law.properties.customerId
        sharedKey: law.listKeys().primarySharedKey
      }
    }
  }
}

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: acrName
  location: location
  tags: tags
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: false
    publicNetworkAccess: 'Enabled'
  }
}

// App UAMI → AcrPull on the registry
resource raAppAcr 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: acr
  name: guid(acr.id, uami.id, roleAcrPull)
  properties: {
    principalId: uami.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: roleAcrPull
  }
}

// App UAMI → Cognitive Services OpenAI User (synthesize_answer call on Tab 3)
resource raAppOpenAI 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: openai
  name: guid(openai.id, uami.id, roleCognitiveServicesOpenAIUser)
  properties: {
    principalId: uami.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: roleCognitiveServicesOpenAIUser
  }
}

// App UAMI → Search Index Data Reader (queries from Tabs 1–3)
resource raAppSearch 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: search
  name: guid(search.id, uami.id, roleSearchIndexDataReader)
  properties: {
    principalId: uami.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: roleSearchIndexDataReader
  }
}

resource webApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: webAppName
  location: location
  tags: union(tags, {
    'azd-service-name': 'web'
  })
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${uami.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: acaEnv.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
        allowInsecure: false
      }
      registries: [
        {
          server: acr.properties.loginServer
          identity: uami.id
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'web'
          image: webImage
          resources: {
            cpu: json(webCpu)
            memory: webMemory
          }
          env: [
            { name: 'AZURE_CLIENT_ID',              value: uami.properties.clientId }
            { name: 'SEARCH_ENDPOINT',              value: 'https://${search.name}.search.windows.net' }
            { name: 'SEARCH_INDEX_NAME',            value: 'utility-ops' }
            { name: 'AOAI_ENDPOINT',                value: openai.properties.endpoint }
            { name: 'AOAI_EMBEDDING_DEPLOYMENT',    value: embeddingDeployment.name }
            { name: 'AOAI_EMBEDDING_MODEL',         value: embeddingModelName }
            { name: 'AOAI_EMBEDDING_DIMENSIONS',    value: '3072' }
            { name: 'AOAI_CHAT_DEPLOYMENT',         value: chatDeployment.name }
            { name: 'AOAI_CHAT_MODEL',              value: chatModelName }
            { name: 'KNOWLEDGE_SOURCE_NAME',        value: 'utility-knowledge-source' }
            { name: 'KNOWLEDGE_BASE_NAME',          value: 'utility-knowledge-base' }
            { name: 'PUBLIC_DEMO',                  value: 'true' }
          ]
        }
      ]
      scale: {
        minReplicas: webMinReplicas
        maxReplicas: webMaxReplicas
      }
    }
  }
  dependsOn: [
    raAppAcr
    raAppOpenAI
    raAppSearch
  ]
}

// -----------------------------------------------------------------------------
// Outputs — map directly onto .env.example
// -----------------------------------------------------------------------------

output SEARCH_ENDPOINT string = 'https://${search.name}.search.windows.net'
output SEARCH_INDEX_NAME string = 'utility-ops'
output AOAI_ENDPOINT string = openai.properties.endpoint
output AOAI_EMBEDDING_DEPLOYMENT string = embeddingDeployment.name
output AOAI_EMBEDDING_MODEL string = embeddingModelName
output AOAI_EMBEDDING_DIMENSIONS int = 3072
output AOAI_CHAT_DEPLOYMENT string = chatDeployment.name
output AOAI_CHAT_MODEL string = chatModelName
output KNOWLEDGE_SOURCE_NAME string = 'utility-knowledge-source'
output KNOWLEDGE_BASE_NAME string = 'utility-knowledge-base'

output searchServiceName string = search.name
output openaiAccountName string = openai.name
output resourceGroupName string = resourceGroup().name
output location string = location
output searchLocation string = searchLocation

// azd integration outputs — required by `azd deploy` and consumed by Container App pipelines
output AZURE_CONTAINER_REGISTRY_ENDPOINT string = acr.properties.loginServer
output AZURE_CONTAINER_REGISTRY_NAME string = acr.name
output AZURE_RESOURCE_GROUP string = resourceGroup().name
output WEB_APP_NAME string = webApp.name
output WEB_URI string = 'https://${webApp.properties.configuration.ingress.fqdn}'
