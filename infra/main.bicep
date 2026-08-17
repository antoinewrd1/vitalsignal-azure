// VitalSignal Azure footprint.
// Deploy:  az deployment group create -g rg-vitalsignal -f infra/main.bicep \
//            -p adminObjectId=<your-aad-object-id>
//
// Scope notes (deliberate):
//  * Databricks workspace is created here, but its CLUSTERS and JOBS are not --
//    those belong to the Databricks REST API / bundle (orchestration/), because
//    Bicep cannot reach inside the workspace control plane.
//  * Role assignments below implement identity-based access: the Databricks
//    managed identity reads/writes the lake; nobody handles storage keys.
//  * Azure OpenAI model *deployments* are included because capacity is the part
//    people forget; the account alone serves nothing.

@description('Deployment location')
param location string = resourceGroup().location

@description('Object ID of the operator (for Key Vault + Storage data-plane roles)')
param adminObjectId string

@description('Base name for all resources')
param baseName string = 'vitalsignal'

var suffix = uniqueString(resourceGroup().id)
var storageName = 'st${baseName}${take(suffix, 6)}'

// ---------------- ADLS Gen2 lakehouse ----------------
resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageName
  location: location
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    isHnsEnabled: true            // hierarchical namespace = ADLS Gen2
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
  }
}

resource lakeFs 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  name: '${storage.name}/default/lakehouse'
  properties: {}
}

// ---------------- Databricks ----------------
resource databricks 'Microsoft.Databricks/workspaces@2024-05-01' = {
  name: 'dbw-${baseName}'
  location: location
  sku: { name: 'premium' }        // premium => Unity Catalog + cluster policies
  properties: {
    managedResourceGroupId: subscriptionResourceId(
      'Microsoft.Resources/resourceGroups', 'rg-${baseName}-dbw-managed'
    )
  }
}

// Databricks Access Connector: the managed identity Unity Catalog uses to
// reach the lake. This is the "no storage keys anywhere" piece.
resource accessConnector 'Microsoft.Databricks/accessConnectors@2024-05-01' = {
  name: 'dbac-${baseName}'
  location: location
  identity: { type: 'SystemAssigned' }
}

var blobContributor = subscriptionResourceId(
  'Microsoft.Authorization/roleDefinitions',
  'ba92f5b4-2d11-453d-a403-e96b0029c9fe' // Storage Blob Data Contributor
)

resource lakeRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storage.id, accessConnector.id, blobContributor)
  scope: storage
  properties: {
    roleDefinitionId: blobContributor
    principalId: accessConnector.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

resource adminLakeRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storage.id, adminObjectId, blobContributor)
  scope: storage
  properties: {
    roleDefinitionId: blobContributor
    principalId: adminObjectId
    principalType: 'User'
  }
}

// ---------------- Azure ML ----------------
resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: 'appi-${baseName}'
  location: location
  kind: 'web'
  properties: { Application_Type: 'web' }
}

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: 'kv-${baseName}-${take(suffix, 6)}'
  location: location
  properties: {
    sku: { family: 'A', name: 'standard' }
    tenantId: subscription().tenantId
    enableRbacAuthorization: true
  }
}

resource amlWorkspace 'Microsoft.MachineLearningServices/workspaces@2024-04-01' = {
  name: 'mlw-${baseName}'
  location: location
  identity: { type: 'SystemAssigned' }
  properties: {
    storageAccount: storage.id
    keyVault: keyVault.id
    applicationInsights: appInsights.id
  }
}

// ---------------- Azure OpenAI + AI Search ----------------
resource aoai 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: 'aoai-${baseName}'
  location: location
  kind: 'OpenAI'
  sku: { name: 'S0' }
  properties: {
    customSubDomainName: 'aoai-${baseName}-${take(suffix, 6)}'
    disableLocalAuth: true        // Entra ID only; matches the app's auth path
  }
}

resource gpt4oMini 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: aoai
  name: 'gpt-4o-mini'
  sku: { name: 'GlobalStandard', capacity: 30 }
  properties: {
    model: { format: 'OpenAI', name: 'gpt-4o-mini', version: '2024-07-18' }
  }
}

resource search 'Microsoft.Search/searchServices@2024-06-01-preview' = {
  name: 'srch-${baseName}-${take(suffix, 6)}'
  location: location
  sku: { name: 'basic' }
  properties: { replicaCount: 1, partitionCount: 1 }
}

output storageAccountName string = storage.name
output databricksWorkspaceUrl string = databricks.properties.workspaceUrl
output amlWorkspaceName string = amlWorkspace.name
output aoaiEndpoint string = aoai.properties.endpoint
output searchEndpoint string = 'https://${search.name}.search.windows.net'
